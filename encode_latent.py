import torch
import torch.multiprocessing as mp
from diffusers import AudioLDM2Pipeline
import torchaudio
import librosa
import os
import numpy as np
from tqdm import tqdm
import traceback
from audioldm2.utils import default_audioldm_config
from audioldm2.utilities.audio.stft import TacotronSTFT
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# ==========================================================================================
# Original audio processing and VAE encode/decode code (Unchanged as requested)
# ==========================================================================================

def get_mel_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    # audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, magnitudes, phases, energy = _stft.mel_spectrogram(audio)
    # melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    # magnitudes = torch.squeeze(magnitudes, 0).numpy().astype(np.float32)
    # energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
    return melspec.squeeze(0), magnitudes.squeeze(0), energy.squeeze(0)

def pad_wav(waveform, segment_length):
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length
    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    elif waveform_length < segment_length:
        temp_wav = np.zeros((1, segment_length))
        temp_wav[:, :waveform_length] = waveform
    return temp_wav

def _pad_spec(fbank, target_length=1024):
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    if fbank.size(-1) % 2 != 0:
        fbank = fbank[..., :-1]
    return fbank

def normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5

def read_wav_file(filename, segment_length):
    waveform, sr = torchaudio.load(filename, format="mp4")
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    waveform = waveform.numpy()[0, ...]
    waveform = normalize_wav(waveform)
    waveform = waveform[None, ...]
    waveform = pad_wav(waveform, segment_length)
    return waveform

def wav_to_fbank(filename, target_length=1024, fn_STFT=None):
    assert fn_STFT is not None
    waveform = read_wav_file(filename, target_length * 160)
    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)
    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)
    fbank = torch.FloatTensor(fbank.T)
    log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)
    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )
    return fbank, log_magnitudes_stft, waveform

def encode_audio_from_video(video_path, vae, fn_STFT, device):
    try:
        config=default_audioldm_config()
        duration=3
        mel, _, _ = wav_to_fbank(
            video_path, target_length=int(duration * 102.4), fn_STFT=fn_STFT
        )
        mel=mel.unsqueeze(0).unsqueeze(0).to(device).to(torch.float16)
        with torch.no_grad():
            latent_representation  = vae.encode(mel).latent_dist.mode()
        return latent_representation.cpu().numpy(), "SUCCESS"
    except Exception as e:
        error_message = f"处理文件 '{os.path.basename(video_path)}' 时发生错误: {e}\n{traceback.format_exc()}"
        return None, error_message

# ==========================================================================================
# DDP Setup and Data Handling
# ==========================================================================================

class VideoDataset(Dataset):
    """Custom Dataset to handle video file paths."""
    def __init__(self, file_list, input_dir, output_dir):
        self.file_list = file_list
        self.input_dir = input_dir
        self.output_dir = output_dir

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        video_path = os.path.join(self.input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(self.output_dir, f"{base_name}.npy")
        return video_path, output_path

def setup_ddp(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """Destroys the distributed process group."""
    dist.destroy_process_group()

def setup_audioldm2_vae(gpu_id, repo_id="cvssp/audioldm2", torch_dtype=torch.float16):
    """Loads and sets up the AudioLDM 2 VAE model on the specified GPU."""
    device = f"cuda:{gpu_id}"
    if gpu_id == 0:
        print(f"正在加载 AudioLDM 2 模型...")
    # Note: DDP doesn't require wrapping the model if you're only doing inference
    # and not synchronizing gradients. Each process will have its own model copy.
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch_dtype, resume_download=True)
    pipe = pipe.to(device)
    if gpu_id == 0:
        print(f"模型已成功加载。")
    return pipe.vae, pipe.feature_extractor, device

def main_worker(rank, world_size, input_dir, output_dir):
    """
    The main worker function for each DDP process.
    Handles model setup, data loading, and processing for its assigned data split.
    """
    setup_ddp(rank, world_size)
    
    # --- 1. In-process model setup ---
    try:
        vae, _, device = setup_audioldm2_vae(rank)
    except Exception as e:
        print(f"[GPU-{rank}]: 模型加载失败: {e}")
        cleanup_ddp()
        return

    # --- 2. Prepare DataLoader with DistributedSampler ---
    # Only the main process should scan the directory
    all_files = []
    if rank == 0:
        all_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")])
        if not all_files:
            print(f"在目录 '{input_dir}' 中没有找到 .mp4 文件。")
    
    # Broadcast the list of files from rank 0 to all other processes
    # This ensures every process has the same file list to create the dataset
    file_list_tensor = torch.tensor(bytearray(str.encode(','.join(all_files))), dtype=torch.uint8).to(rank)
    if rank == 0:
        size_tensor = torch.tensor([file_list_tensor.shape[0]], dtype=torch.long).to(rank)
    else:
        size_tensor = torch.tensor([0], dtype=torch.long).to(rank)

    dist.broadcast(size_tensor, src=0)
    
    # Resize tensor on non-zero ranks
    if rank != 0:
        file_list_tensor.resize_(size_tensor.item())

    dist.broadcast(file_list_tensor, src=0)
    
    all_files_str = file_list_tensor.cpu().numpy().tobytes().decode()
    all_files = all_files_str.split(',') if all_files_str else []

    if not all_files:
        cleanup_ddp()
        return

    dataset = VideoDataset(all_files, input_dir, output_dir)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    # Batch size is 1 since we process one file at a time. num_workers can be increased.
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=64, pin_memory=True, sampler=sampler)

    # --- 3. Setup STFT utility ---
    config = default_audioldm_config()
    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    # --- 4. Process files assigned by the DataLoader ---
    # The progress bar is only shown on the main process (rank 0)
    if rank == 0:
        progress_bar = tqdm(total=len(dataset), desc="全体处理中")

    for i, (video_path_tuple, output_path_tuple) in enumerate(dataloader):
        video_path = video_path_tuple[0]
        output_path = output_path_tuple[0]

        if os.path.exists(output_path):
            if rank == 0:
                progress_bar.update(1) # Still update progress for skipped files
            continue

        latent_np, status = encode_audio_from_video(video_path, vae, fn_STFT, device)

        if status == "SUCCESS" and latent_np is not None:
            np.save(output_path, latent_np)
        else:
            # Errors are printed to the console from the respective process
            tqdm.write(f"[GPU-{rank} 错误]: {status}")
        
        if rank == 0:
            progress_bar.update(1)

    if rank == 0:
        progress_bar.close()

    cleanup_ddp()


def batch_process_videos_ddp(input_dir, output_dir):
    """
    Main function: responsible for task distribution and launching DDP processes.
    """
    if not torch.cuda.is_available():
        print("错误：未检测到CUDA设备。请在单卡CPU/GPU模式下运行。")
        return

    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 块可用的GPU。")

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")
        except OSError as e:
            print(f"创建输出目录失败: {e}")
            return
    
    # Use mp.spawn to launch DDP processes
    mp.spawn(main_worker,
            args=(world_size, input_dir, output_dir),
            nprocs=world_size,
            join=True)

    print("\n🎉 所有视频处理完成！")


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    video_directory_list=["vggsound_05_3s","vggsound_06_3s","vggsound_07_3s","vggsound_08_3s","vggsound_09_3s","vggsound_10_3s","vggsound_11_3s","vggsound_12_3s","vggsound_13_3s","vggsound_14_3s"]
    input_video_directory_base="/blob/vggsound_cropped"
    output_latent_directory_base = "/blob/vggsound_cropped_audio_latent"
    
    for video_dir in video_directory_list:
        input_video_directory=os.path.join(input_video_directory_base,video_dir)
        output_latent_directory=os.path.join(output_latent_directory_base,video_dir)

        print(f"\n--- 开始处理目录: {input_video_directory} ---")
        try:
            batch_process_videos_ddp(input_video_directory, output_latent_directory)
        except Exception as e:
            print(f"\n程序运行期间发生严重错误: {e}")
            print(traceback.format_exc())