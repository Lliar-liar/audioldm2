import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from diffusers import AudioLDM2Pipeline
import torchaudio
import os
import numpy as np
from tqdm import tqdm
import traceback
from audioldm2.utils import default_audioldm_config
from audioldm2.utilities.audio.stft import TacotronSTFT

# =====================================================================================
#  可调优的性能参数
# =====================================================================================
# BATCH_SIZE: 一次性送入GPU处理的文件数量。根据你的GPU显存调整。
#             对于3秒音频，32或64是比较合理的值。显存越大，可以设置得越高。
BATCH_SIZE = 32

# NUM_WORKERS: 在后台加载和预处理数据的子进程数量。
#              建议设置为你服务器CPU核心数的一半左右，例如，如果你有16核，可以设为8。
#              设置为0表示只使用主进程加载数据（会变慢）。
NUM_WORKERS = 32
# =====================================================================================


# =====================================================================================
#  1. 音频预处理的辅助函数 (基本保持不变，但增强了健壮性)
# =====================================================================================

def get_mel_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, magnitudes, phases, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    return melspec

def pad_wav(waveform, segment_length):
    waveform_length = waveform.shape[-1]
    if waveform_length > segment_length:
        return waveform[:, :segment_length]
    elif waveform_length < segment_length:
        temp_wav = np.zeros((1, segment_length))
        temp_wav[:, :waveform_length] = waveform
        return temp_wav
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
    if np.max(np.abs(waveform)) < 1e-6:
        return waveform
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5

def read_wav_file(filename, segment_length):
    try:
        waveform, sr = torchaudio.load(filename, format="mp4", backend="ffmpeg")
        if waveform.numel() == 0: # 检查是否为空音频
            raise ValueError("Loaded waveform is empty.")
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        waveform = waveform.numpy()
        waveform = normalize_wav(waveform)
        waveform = pad_wav(waveform, segment_length)
    except Exception as e:
        # tqdm.write(f"Warning: Failed to load {os.path.basename(filename)}. Error: {e}. Returning silence.")
        waveform = np.zeros((1, segment_length))
    return waveform

def wav_to_fbank(filename, target_length, fn_STFT):
    hop_size = 160  # From default config
    waveform = read_wav_file(filename, target_length * hop_size)
    waveform = waveform[0, ...]
    fbank = get_mel_from_wav(waveform, fn_STFT)
    fbank = torch.FloatTensor(fbank.T)
    fbank = _pad_spec(fbank, target_length)
    return fbank

# =====================================================================================
#  2. 优化的核心：自定义 Dataset
# =====================================================================================

class VideoAudioDataset(Dataset):
    """
    这个类负责高效地加载和预处理单个音频文件。
    DataLoader将使用这个类，并利用多进程（num_workers）来并行执行这里的操作。
    """
    def __init__(self, file_list, input_dir, duration=3):
        self.input_dir = input_dir
        self.file_list = file_list
        self.target_length = int(duration * 102.4)

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
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        video_path = os.path.join(self.input_dir, filename)
        
        try:
            mel = wav_to_fbank(video_path, self.target_length, self.fn_STFT)
        except Exception:
            # 如果某个文件处理失败，返回一个全零的张量，以保证批次处理不中断
            mel = torch.zeros((self.target_length, 80)) # 80 is n_mel_channels
        
        return mel, video_path

# =====================================================================================
#  3. 模型加载与优化的工作进程函数
# =====================================================================================

def setup_audioldm2_vae(gpu_id, repo_id="cvssp/audioldm2", torch_dtype=torch.float16):
    device = f"cuda:{gpu_id}"
    print(f"[GPU-{gpu_id}]: 正在加载 AudioLDM 2 VAE 到 {device}...")
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch_dtype, resume_download=True)
    pipe = pipe.to(device)
    print(f"[GPU-{gpu_id}]: 模型已成功加载。")
    return pipe.vae, device

def process_files_on_gpu_optimized(gpu_id, file_chunk, input_dir, output_dir):
    """
    这是由单个GPU进程执行的工作函数。
    它内部使用DataLoader来创建数据加载流水线。
    """
    try:
        vae, device = setup_audioldm2_vae(gpu_id)
    except Exception as e:
        print(f"[GPU-{gpu_id}]: 模型加载失败: {e}")
        return len(file_chunk), 0

    dataset = VideoAudioDataset(file_chunk, input_dir)
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=False
    )
    
    success_count = 0
    error_count = 0
    
    progress_bar = tqdm(data_loader, desc=f"GPU-{gpu_id} 处理中", position=gpu_id, leave=True)
    
    for mel_batch, path_batch in progress_bar:
        try:
            final_mel_batch = []
            final_output_paths = []

            # 过滤掉已经存在的文件，避免重复计算
            for i, video_path in enumerate(path_batch):
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}.npy")
                if not os.path.exists(output_path):
                    final_mel_batch.append(mel_batch[i])
                    final_output_paths.append(output_path)
            
            if not final_mel_batch:
                success_count += len(path_batch)
                continue

            # 将需要处理的数据合并成一个新的批次
            mel_to_process = torch.stack(final_mel_batch)
            mel_to_process = mel_to_process.unsqueeze(1).to(device, dtype=torch.float16)
            
            with torch.no_grad():
                latent_batch = vae.encode(mel_to_process).latent_dist.mode()
            
            latent_batch_np = latent_batch.cpu().numpy()

            for latent_np, output_path in zip(latent_batch_np, final_output_paths):
                np.save(output_path, latent_np)
            
            success_count += len(path_batch) # 将跳过和已处理的都算作成功

        except Exception as e:
            tqdm.write(f"[GPU-{gpu_id} 错误]: 批处理失败，跳过此批次。错误: {e}")
            error_count += len(path_batch)

    return error_count, success_count

# =====================================================================================
#  4. 主函数和任务分发
# =====================================================================================

def batch_process_videos_multi_gpu(input_dir, output_dir):
    if not torch.cuda.is_available():
        print("错误：未检测到CUDA设备。")
        return

    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 块可用的GPU。")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    all_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".mp4", ".wav", ".flac", ".m4a"))])
    if not all_files:
        print(f"在目录 '{input_dir}' 中没有找到支持的音/视频文件。")
        return
        
    print(f"在输入目录中找到 {len(all_files)} 个文件。准备分发任务...")

    file_chunks = np.array_split(all_files, num_gpus)
    
    tasks = []
    for gpu_id, chunk in enumerate(file_chunks):
        if len(chunk) > 0:
            tasks.append((gpu_id, list(chunk), input_dir, output_dir))
            print(f"  -> GPU-{gpu_id} 将处理 {len(chunk)} 个文件。")

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=len(tasks)) as pool:
        results = pool.starmap(process_files_on_gpu_optimized, tasks)

    total_errors = sum([res[0] for res in results])
    total_success = sum([res[1] for res in results])

    print("\n" + "="*50)
    print("--- 处理完成，生成报告 ---")
    print(f"总计成功处理: {total_success} 个文件")
    print(f"总计处理失败: {total_errors} 个文件")
    print("="*50 + "\n")


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
        print("多进程启动方法已设置为 'spawn'。")
    except RuntimeError:
        print("启动方法已经被设置，忽略。")

    # =======================================================================
    #  在这里配置你的输入输出目录
    # =======================================================================
    video_directory_list = [
        "vggsound_00_3s", "vggsound_01_3s", "vggsound_02_3s", "vggsound_03_3s", 
        "vggsound_04_3s", "vggsound_05_3s", "vggsound_06_3s", "vggsound_07_3s", 
        "vggsound_08_3s", "vggsound_09_3s", "vggsound_10_3s", "vggsound_11_3s", 
        "vggsound_12_3s", "vggsound_13_3s", "vggsound_14_3s"
    ]
    input_video_directory_base = "/blob/vggsound_cropped"
    output_latent_directory_base = "/blob/vggsound_cropped_audio_latent"
    # =======================================================================

    for video_dir in video_directory_list:
        input_video_directory = os.path.join(input_video_directory_base, video_dir)
        output_latent_directory = os.path.join(output_latent_directory_base, video_dir)

        print(f"\n{'='*20} 开始处理目录: {video_dir} {'='*20}")
        print(f"输入: {input_video_directory}")
        print(f"输出: {output_latent_directory}")

        if not os.path.isdir(input_video_directory):
            print(f"警告: 输入目录不存在，跳过: {input_video_directory}")
            continue

        try:
            batch_process_videos_multi_gpu(input_video_directory, output_latent_directory)
        except Exception as e:
            print(f"\n处理目录 '{video_dir}' 时发生严重错误: {e}")
            print(traceback.format_exc())

    print("\n🎉 所有任务处理完成！")