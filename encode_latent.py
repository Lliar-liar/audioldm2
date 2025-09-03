import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from diffusers import AudioLDM2Pipeline
import torchaudio
import numpy as np
from tqdm import tqdm
import traceback

from audioldm2.utils import default_audioldm_config
from audioldm2.utilities.audio.stft import TacotronSTFT

# ==============================================================================
# 1. HELPER FUNCTIONS (Audio Preprocessing)
# These are mostly unchanged but are required for the script to be self-contained.
# ==============================================================================

def get_mel_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, _, _, _ = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    return melspec

def pad_wav(waveform, segment_length):
    waveform_length = waveform.shape[-1]
    if waveform_length > 100:
        if waveform_length > segment_length:
            return waveform[:, :segment_length]
        elif waveform_length < segment_length:
            temp_wav = np.zeros((1, segment_length))
            temp_wav[:, :waveform_length] = waveform
            return temp_wav
    else: # Waveform is too short or silent
        return np.zeros((1, segment_length))
    return waveform

def _pad_spec(fbank, target_length=1024):
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    return fbank

def normalize_wav(waveform):
    if np.max(np.abs(waveform)) > 1e-6:
        waveform = waveform - np.mean(waveform)
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        return waveform * 0.5
    return waveform # Return zeros if silent

def read_wav_file(filename, segment_length):
    try:
        waveform, sr = torchaudio.load(filename, format="mp4")
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        waveform = waveform.numpy()
        waveform = normalize_wav(waveform)
        waveform = pad_wav(waveform, segment_length)
    except Exception:
        # If any error occurs during loading, return a silent waveform
        waveform = np.zeros((1, segment_length))
    return waveform

def wav_to_fbank(filename, target_length, fn_STFT):

    waveform = read_wav_file(filename, target_length * 160)
    waveform = torch.FloatTensor(waveform[0, ...])
    fbank = get_mel_from_wav(waveform, fn_STFT)
    fbank = torch.FloatTensor(fbank.T)
    fbank = _pad_spec(fbank, target_length)
    return fbank

# ==============================================================================
# 2. PYTORCH DATASET
# This class efficiently loads and preprocesses one audio file at a time.
# The DataLoader will use this in multiple worker processes in the background.
# ==============================================================================

class VideoAudioDataset(Dataset):
    def __init__(self, file_paths, input_dir, target_length_sec=3):
        self.input_dir = input_dir
        self.file_paths = file_paths
        self.target_length = int(target_length_sec * 102.4)

        config = default_audioldm_config()
        self.fn_STFT = TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        full_path = os.path.join(self.input_dir, filepath)
        
        # The output filename is derived from the input
        output_filename = f"{os.path.splitext(filepath)[0]}.npy"
        
        mel = wav_to_fbank(
            full_path, target_length=self.target_length, fn_STFT=self.fn_STFT
        )
        return mel, output_filename

# ==============================================================================
# 3. DDP SETUP AND MAIN WORKER FUNCTION
# This is the core logic that will be run on each GPU.
# ==============================================================================

def setup(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def run_worker(rank, world_size, all_files, input_dir, output_dir, batch_size):
    """
    The main function executed by each GPU process.
    `rank` is the GPU ID (from 0 to world_size-1).
    """
    print(f"Starting DDP worker on GPU {rank}.")
    setup(rank, world_size)

    # --- Model Loading ---
    # Each process loads its own copy of the model into its assigned GPU
    pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=torch.float16)
    vae = pipe.vae.to(rank)
    # Wrap the model with DDP
    vae_ddp = DDP(vae, device_ids=[rank])
    vae_ddp.eval()

    # --- Data Loading ---
    dataset = VideoAudioDataset(all_files, input_dir)
    # The sampler ensures each GPU gets a different slice of the data
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    # `num_workers` creates background processes for data loading, crucial for performance
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=64, # Adjust based on your CPU cores
        pin_memory=True,
    )
    
    # --- Processing Loop ---
    # Only the main process (rank 0) shows the progress bar
    if rank == 0:
        progress_bar = tqdm(data_loader, desc="All GPUs Processing")
    else:
        progress_bar = data_loader

    with torch.no_grad():
        for mel_batch, output_filenames in progress_bar:
            # Check for already processed files
            valid_indices = []
            final_output_paths = []
            for i, fname in enumerate(output_filenames):
                output_path = os.path.join(output_dir, fname)
                if not os.path.exists(output_path):
                    valid_indices.append(i)
                    final_output_paths.append(output_path)
            
            if not valid_indices:
                continue # Skip batch if all files already exist

            # Filter the batch to only include unprocessed files
            mel_batch = mel_batch[valid_indices]
            
            # Move data to the current GPU
            mel_batch = mel_batch.unsqueeze(1).to(rank, dtype=torch.float16)
            
            # Perform inference. Use `.module` to access the original model's methods
            latent_dist = vae_ddp.module.encode(mel_batch).latent_dist
            latent_batch = latent_dist.mode().cpu().numpy()

            # Save the results
            for latent_np, output_path in zip(latent_batch, final_output_paths):
                np.save(output_path, latent_np)
    
    cleanup()
    if rank == 0:
        print(f"\nGPU {rank} finished.")

# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# This sets up the environment and spawns the DDP workers.
# ==============================================================================

def main():
    # --- Configuration ---
    video_directory_list=["vggsound_00_3s"]
    # video_directory_list = [
    #     "vggsound_05_3s", "vggsound_06_3s", "vggsound_07_3s", "vggsound_08_3s",
    #     "vggsound_09_3s", "vggsound_10_3s", "vggsound_11_3s", "vggsound_12_3s",
    #     "vggsound_13_3s", "vggsound_14_s"
    # ]
    input_video_directory_base = "/blob/vggsound_cropped"
    output_latent_directory_base = "/blob/vggsound_cropped_audio_latent"
    
    # --- Tunable Parameters ---
    # Per-GPU batch size. Total batch size will be (batch_size * num_gpus)
    # Adjust based on your VRAM. 32 or 64 is a good starting point.
    batch_size = 16
    
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires multiple GPUs.")
        return
        
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs. Starting DDP processing.")

    for video_dir in video_directory_list:
        input_dir = os.path.join(input_video_directory_base, video_dir)
        output_dir = os.path.join(output_latent_directory_base, video_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        all_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")])
        if not all_files:
            print(f"No .mp4 files found in {input_dir}. Skipping.")
            continue
            
        print(f"\nProcessing {len(all_files)} files from {input_dir}...")

        try:
            # mp.spawn launches `world_size` processes, each running `run_worker`
            mp.spawn(
                run_worker,
                args=(world_size, all_files, input_dir, output_dir, batch_size),
                nprocs=world_size,
                join=True
            )
            print(f"Finished processing directory: {video_dir}")
        except Exception as e:
            print(f"\nAn error occurred during processing directory {video_dir}: {e}")
            print(traceback.format_exc())

    print("\n🎉 All video directories processed successfully!")


if __name__ == '__main__':
    # 'spawn' start method is recommended for CUDA
    # In DDP with mp.spawn, this is handled automatically, but setting it doesn't hurt.
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()