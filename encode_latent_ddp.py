import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from diffusers import AudioLDM2Pipeline
from diffusers import AutoencoderKL
import torchaudio
import librosa
import os
import numpy as np
from tqdm import tqdm
import traceback
from audioldm2.utils import default_audioldm_config
from audioldm2.utilities.audio.stft import TacotronSTFT
from diffusers import AutoencoderKL

# ========== 修改的音频处理函数以支持批处理 ==========

def get_mel_from_wav_batch(audio_batch, _stft):
    """处理一批音频"""
    # audio_batch: [batch_size, samples] 或 [batch_size, 1, samples]
    # if audio_batch.dim() == 2:
    #     audio_batch = audio_batch.unsqueeze(1)  # [batch_size, 1, samples]
    # elif audio_batch.dim() == 3 and audio_batch.size(1) == 1:
    #     audio_batch = audio_batch.squeeze(1)  # 确保是 [batch_size, samples]
    # print(audio_batch.shape)
    
    audio_batch = torch.clip(audio_batch, -1, 1)
    audio_batch = torch.autograd.Variable(audio_batch, requires_grad=False)
    
    # 批量处理
    melspec, magnitudes, phases, energy = _stft.mel_spectrogram(audio_batch)
    
    return melspec, magnitudes, energy

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
    # cut and pad
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
    try:
        waveform, sr = torchaudio.load(filename, format="mp4", backend="ffmpeg")
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        waveform = waveform.numpy()[0, ...]
        waveform = normalize_wav(waveform)
        waveform = waveform[None, ...]
        waveform = pad_wav(waveform, segment_length)

        max_val = np.max(np.abs(waveform))
        if max_val > 1e-8:
            waveform = waveform / max_val
            waveform = 0.5 * waveform
        return waveform
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def wav_to_fbank_batch(filenames, target_length=1024, fn_STFT=None, device=None):
    """批量处理多个音频文件"""
    assert fn_STFT is not None
    
    batch_waveforms = []
    valid_indices = []
    
    # 读取所有音频文件
    for i, filename in enumerate(filenames):
        waveform = read_wav_file(filename, target_length * 160)
        if waveform is not None:
            waveform = waveform[0, ...]
            batch_waveforms.append(waveform)
            valid_indices.append(i)
    
    if not batch_waveforms:
        return None, None, None, []
    
    # 转换为批量tensor
    batch_waveforms = torch.FloatTensor(np.stack(batch_waveforms)).to(device)
    
    # 批量处理mel spectrogram
    fbank_batch, log_magnitudes_stft_batch, energy_batch = get_mel_from_wav_batch(
        batch_waveforms, fn_STFT
    )
    
    # 处理每个样本的padding
    processed_fbanks = []
    processed_log_mags = []
    
    for i in range(fbank_batch.size(0)):
        fbank = fbank_batch[i].T
        log_mag = log_magnitudes_stft_batch[i].T
        
        fbank = _pad_spec(fbank, target_length)
        log_mag = _pad_spec(log_mag, target_length)
        
        processed_fbanks.append(fbank)
        processed_log_mags.append(log_mag)
    
    # Stack回batch
    fbank_batch = torch.stack(processed_fbanks)
    log_magnitudes_stft_batch = torch.stack(processed_log_mags)
    
    return fbank_batch, log_magnitudes_stft_batch, batch_waveforms, valid_indices

def encode_audio_batch_from_videos(video_paths, vae, fn_STFT, device, batch_size=8):
    """批量编码多个视频的音频"""
    try:
        config = default_audioldm_config()
        duration = 3
        
        # 批量处理音频
        mel_batch, _, _, valid_indices = wav_to_fbank_batch(
            video_paths, 
            target_length=int(duration * 102.4), 
            fn_STFT=fn_STFT, 
            device=device
        )
        
        if mel_batch is None:
            return [None] * len(video_paths), ["FAILED"] * len(video_paths)
        
        # mel_batch: [batch_size, freq, time]
        mel_batch = mel_batch.unsqueeze(1).to(torch.float16)  # [batch_size, 1, freq, time]
        
        # 批量编码
        with torch.no_grad():
            latent_representations = vae.encode(mel_batch).latent_dist.mode()
        
        # 准备结果
        results = [None] * len(video_paths)
        statuses = ["FAILED"] * len(video_paths)
        
        # 填充成功的结果
        for i, valid_idx in enumerate(valid_indices):
            results[valid_idx] = latent_representations[i].cpu().numpy()
            statuses[valid_idx] = "SUCCESS"
        
        return results, statuses
        
    except Exception as e:
        error_message = f"批处理时发生错误: {e}\n{traceback.format_exc()}"
        print(error_message)
        return [None] * len(video_paths), ["FAILED"] * len(video_paths)

# ========== 修改的Dataset类以支持批处理 ==========

class VideoDataset(Dataset):
    """简单的数据集类用于加载视频文件路径"""
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")])
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        video_path = os.path.join(self.input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(self.output_dir, f"{base_name}.npy")
        return video_path, output_path

def collate_fn(batch):
    """自定义collate函数来处理批量数据"""
    video_paths = [item[0] for item in batch]
    output_paths = [item[1] for item in batch]
    return video_paths, output_paths

def setup_ddp(rank, world_size):
    """初始化DDP环境"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """清理DDP"""
    dist.destroy_process_group()

def setup_audioldm2_vae_ddp(rank, repo_id="cvssp/audioldm2", torch_dtype=torch.float16):
    """为DDP加载并设置AudioLDM 2 VAE模型"""
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print(f"[Rank {rank}]: 正在加载 AudioLDM 2 模型...")
    vae = AutoencoderKL.from_pretrained(
        repo_id, subfolder="vae", torch_dtype=torch_dtype, resume_download=True
    )
    vae = vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    
    if rank == 0:
        print(f"[Rank {rank}]: 模型已成功加载。")
    
    return vae, None, device

def process_batch_ddp(rank, world_size, input_dir, output_dir, batch_size=8):
    """DDP工作函数，处理分配给当前rank的数据"""
    # 设置DDP
    setup_ddp(rank, world_size)
    
    try:
        # 加载模型
        vae, _, device = setup_audioldm2_vae_ddp(rank)
        
        # 创建数据集和分布式采样器
        dataset = VideoDataset(input_dir, output_dir)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        # 使用更大的batch size和自定义collate函数
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=sampler, 
            num_workers=2,  # 增加数据加载线程
            collate_fn=collate_fn,
            pin_memory=True  # 加速数据传输到GPU
        )
        
        # 设置STFT
        config = default_audioldm_config()
        fn_STFT = TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        ).to(device)
        
        error_count = 0
        success_count = 0
        
        # 计算总批次数
        total_batches = len(dataloader)
        
        # 只在rank 0显示进度条
        iterator = tqdm(dataloader, desc=f"Rank {rank} 处理中 (batch_size={batch_size})") if rank == 0 else dataloader
        
        for batch_idx, (video_paths, output_paths) in enumerate(iterator):
            # 批量编码
            latent_results, statuses = encode_audio_batch_from_videos(
                video_paths, vae, fn_STFT, device, batch_size
            )
            
            # 保存结果
            for i, (latent_np, status, output_path) in enumerate(zip(latent_results, statuses, output_paths)):
                if status == "SUCCESS" and latent_np is not None:
                    np.save(output_path, latent_np)
                    success_count += 1
                else:
                    if rank == 0:
                        print(f"[Rank {rank} 错误]: 文件 {os.path.basename(video_paths[i])} 处理失败")
                    error_count += 1
            
            # 定期清理缓存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # 收集所有rank的统计信息
        error_tensor = torch.tensor([error_count], device=device)
        success_tensor = torch.tensor([success_count], device=device)
        
        dist.all_reduce(error_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(success_tensor, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            print(f"\n总计成功处理: {success_tensor.item()} 个文件")
            print(f"总计处理失败: {error_tensor.item()} 个文件")
            
    except Exception as e:
        print(f"[Rank {rank}]: 处理过程中发生错误: {e}")
        traceback.print_exc()
    finally:
        if 'vae' in locals():
            del vae
        torch.cuda.empty_cache()
        cleanup()

def batch_process_videos_ddp(input_dir, output_dir, batch_size=8):
    """主函数：使用DDP处理视频"""
    # 检查GPU数量
    if not torch.cuda.is_available():
        print("错误：未检测到CUDA设备。")
        return
    
    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 块可用的GPU，将使用DDP进行处理。")
    print(f"每个GPU的batch size: {batch_size}")
    
    # 检查并创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")
    
    # 检查输入文件
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")]
    if not all_files:
        print(f"在目录 '{input_dir}' 中没有找到 .mp4 文件。")
        return
    
    print(f"在输入目录中找到 {len(all_files)} 个 .mp4 文件。")
    
    # 使用spawn启动DDP进程
    mp.spawn(
        process_batch_ddp,
        args=(world_size, input_dir, output_dir, batch_size),
        nprocs=world_size,
        join=True
    )
    torch.cuda.empty_cache()
    print("\n🎉 所有视频处理完成！")

if __name__ == '__main__':
    # 设置多进程启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # 可以根据GPU内存调整batch size
    BATCH_SIZE = 64  # 从8开始，如果内存允许可以增加到16或32
    video_directory_list = ["vggsound_10_3s", "vggsound_11_3s", "vggsound_12_3s", "vggsound_13_3s", "vggsound_14_3s"]
    # video_directory_list = ["vggsound_15_3s", "vggsound_16_3s", "vggsound_17_3s", "vggsound_18_3s", "vggsound_19_3s"]
    input_video_directory_base = "/blob/vggsound_cropped"
    output_latent_directory_base = "/blob/vggsound_cropped_audio_latent_fixed"
   
    for video_dir in video_directory_list:
        print(f"\n处理目录: {video_dir}")
        input_video_directory = os.path.join(input_video_directory_base, video_dir)
        output_latent_directory = os.path.join(output_latent_directory_base, video_dir)
        
        try:
            batch_process_videos_ddp(input_video_directory, output_latent_directory, batch_size=BATCH_SIZE)
        except Exception as e:
            print(f"\n处理 {video_dir} 时发生严重错误: {e}")
            print(traceback.format_exc())
