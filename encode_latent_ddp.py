import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from diffusers import AudioLDM2Pipeline
import torchaudio
import librosa
import os
import numpy as np
from tqdm import tqdm
import traceback
from audioldm2.utils import default_audioldm_config
from audioldm2.utilities.audio.stft import TacotronSTFT

# ========== 音频处理函数保持不变 ==========
def get_mel_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, magnitudes, phases, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    magnitudes = torch.squeeze(magnitudes, 0).numpy().astype(np.float32)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
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
    waveform, sr = torchaudio.load(filename,format="mp4",backend="ffmpeg")
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

# ========== 改进的Dataset类，支持多个目录 ==========
class MultiDirectoryVideoDataset(Dataset):
    """支持多个目录的数据集类"""
    def __init__(self, video_directory_list, input_base_dir, output_base_dir):
        self.files = []
        
        for video_dir in video_directory_list:
            input_dir = os.path.join(input_base_dir, video_dir)
            output_dir = os.path.join(output_base_dir, video_dir)
            
            # 确保输出目录存在
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 收集该目录下的所有文件
            for filename in sorted(os.listdir(input_dir)):
                if filename.lower().endswith(".mp4"):
                    video_path = os.path.join(input_dir, filename)
                    base_name = os.path.splitext(filename)[0]
                    output_path = os.path.join(output_dir, f"{base_name}.npy")
                    self.files.append((video_path, output_path, video_dir))
        
        print(f"总计找到 {len(self.files)} 个视频文件")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        return self.files[idx]

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
        print(f"[Rank {rank}]: 正在加载 AudioLDM 2 模型（仅加载一次）...")
    
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch_dtype, resume_download=True)
    pipe = pipe.to(device)
    
    if rank == 0:
        print(f"[Rank {rank}]: 模型已成功加载。")
    
    return pipe.vae, pipe.feature_extractor, device

def process_all_directories_ddp(rank, world_size, video_directory_list, input_base_dir, output_base_dir):
    """DDP工作函数，一次性处理所有目录"""
    # 设置DDP
    setup_ddp(rank, world_size)
    
    try:
        # 加载模型（只加载一次）
        vae, _, device = setup_audioldm2_vae_ddp(rank)
        
        # 创建包含所有目录的数据集
        dataset = MultiDirectoryVideoDataset(video_directory_list, input_base_dir, output_base_dir)
        
        # 创建分布式采样器
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        # 批次大小设为1，因为视频文件通常较大
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            sampler=sampler, 
            num_workers=2,  # 适度的worker数量
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
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
        )
        
        error_count = 0
        success_count = 0
        current_dir = None
        dir_stats = {}
        
        # 只在rank 0显示进度条
        iterator = tqdm(dataloader, desc=f"Rank {rank} 处理中", total=len(dataloader)) if rank == 0 else dataloader
        
        for batch in iterator:
            video_path, output_path, video_dir = batch
            video_path = video_path[0]  # 解包batch
            output_path = output_path[0]
            video_dir = video_dir[0]
            
            # 检测目录变化，用于显示进度
            if current_dir != video_dir:
                if current_dir is not None and rank == 0:
                    print(f"\n[Rank {rank}] 完成目录: {current_dir}")
                current_dir = video_dir
                if rank == 0:
                    print(f"\n[Rank {rank}] 开始处理目录: {video_dir}")
                
                # 初始化目录统计
                if video_dir not in dir_stats:
                    dir_stats[video_dir] = {'success': 0, 'error': 0}
            
            # 如果文件已存在则跳过
            if os.path.exists(output_path):
                success_count += 1
                dir_stats[video_dir]['success'] += 1
                continue
            
            # 处理文件
            latent_np, status = encode_audio_from_video(video_path, vae, fn_STFT, device)
            
            if status == "SUCCESS" and latent_np is not None:
                try:
                    # 确保目录存在
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    np.save(output_path, latent_np)
                    success_count += 1
                    dir_stats[video_dir]['success'] += 1
                except Exception as e:
                    if rank == 0:
                        print(f"[Rank {rank}] 保存失败: {e}")
                    error_count += 1
                    dir_stats[video_dir]['error'] += 1
            else:
                if rank == 0:
                    print(f"[Rank {rank} 错误]: {status}")
                error_count += 1
                dir_stats[video_dir]['error'] += 1
        
        # 收集所有rank的统计信息
        error_tensor = torch.tensor([error_count], device=device)
        success_tensor = torch.tensor([success_count], device=device)
        
        dist.all_reduce(error_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(success_tensor, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            print("\n" + "="*50)
            print("处理完成，详细统计：")
            print("="*50)
            for video_dir in dir_stats:
                print(f"{video_dir}: 成功 {dir_stats[video_dir]['success']}, 失败 {dir_stats[video_dir]['error']}")
            print("="*50)
            print(f"总计成功处理: {success_tensor.item()} 个文件")
            print(f"总计处理失败: {error_tensor.item()} 个文件")
            
    except Exception as e:
        print(f"[Rank {rank}]: 处理过程中发生错误: {e}")
        traceback.print_exc()
    finally:
        cleanup()

def batch_process_all_videos_ddp(video_directory_list, input_base_dir, output_base_dir):
    """主函数：一次性使用DDP处理所有目录的视频"""
    # 检查GPU数量
    if not torch.cuda.is_available():
        print("错误：未检测到CUDA设备。")
        return
    
    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 块可用的GPU，将使用DDP进行处理。")
    print(f"待处理的目录数量: {len(video_directory_list)}")
    print(f"目录列表: {video_directory_list}")
    
    # 确保基础输出目录存在
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        print(f"已创建输出基础目录: {output_base_dir}")
    
    # 使用spawn启动DDP进程（只启动一次）
    mp.spawn(
        process_all_directories_ddp,
        args=(world_size, video_directory_list, input_base_dir, output_base_dir),
        nprocs=world_size,
        join=True
    )
    
    print("\n🎉 所有目录的视频处理完成！")

if __name__ == '__main__':
    # 设置多进程启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # 所有待处理的目录列表
    video_directory_list = ["vggsound_00_3s","vggsound_01_3s","vggsound_02_3s","vggsound_03_3s","vggsound_04_3s", 
    "vggsound_06_3s", "vggsound_07_3s", "vggsound_08_3s", "vggsound_09_3s",
        "vggsound_10_3s", "vggsound_11_3s", "vggsound_12_3s", "vggsound_13_3s", "vggsound_14_3s",
        "vggsound_15_3s", "vggsound_16_3s", "vggsound_17_3s", "vggsound_18_3s", "vggsound_19_3s",
    ]
    
    input_video_directory_base = "/blob/vggsound_cropped"
    output_latent_directory_base = "/blob/vggsound_cropped_audio_latent_fixed"
    
    # 一次性处理所有目录（模型只加载一次）
    try:
        batch_process_all_videos_ddp(
            video_directory_list, 
            input_video_directory_base, 
            output_latent_directory_base
        )
    except Exception as e:
        print(f"\n程序运行期间发生严重错误: {e}")
        print(traceback.format_exc())
