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
from diffusers import AutoencoderKL
# ========== 以下音频处理函数保持不变 ==========
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
    waveform, sr = torchaudio.load(filename,format="mp4",backend="ffmpeg")
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    waveform = waveform.numpy()[0, ...]
    waveform = normalize_wav(waveform)
    waveform = waveform[None, ...]
    waveform = pad_wav(waveform, segment_length)

    max_val = np.max(np.abs(waveform))
    if max_val > 1e-8:  # 只有在有实际信号时才归一化
        waveform = waveform / max_val
        waveform = 0.5 * waveform
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

# ========== 以下为DDP相关新增/修改的代码 ==========

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
    
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch_dtype, resume_download=True)
    pipe = pipe.to(device)
    
    if rank == 0:
        print(f"[Rank {rank}]: 模型已成功加载。")
    
    return pipe.vae, pipe.feature_extractor, device

def process_batch_ddp(rank, world_size, input_dir, output_dir):
    """DDP工作函数，处理分配给当前rank的数据"""
    # 设置DDP
    setup_ddp(rank, world_size)
    
    try:
        # 加载模型
        vae, _, device = setup_audioldm2_vae_ddp(rank)
        
        # 创建数据集和分布式采样器
        dataset = VideoDataset(input_dir, output_dir)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=64)
        
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
        
        # 只在rank 0显示进度条
        iterator = tqdm(dataloader, desc=f"Rank {rank} 处理中") if rank == 0 else dataloader
        
        for video_path, output_path in iterator:
            video_path = video_path[0]  # 解包batch
            output_path = output_path[0]
            
            # 如果文件已存在则跳过
            # if os.path.exists(output_path):
            #     continue
            
            latent_np, status = encode_audio_from_video(video_path, vae, fn_STFT, device)
            
            if status == "SUCCESS" and latent_np is not None:
                np.save(output_path, latent_np)
                success_count += 1
            else:
                if rank == 0:
                    print(f"[Rank {rank} 错误]: {status}")
                error_count += 1
        
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

def batch_process_videos_ddp(input_dir, output_dir):
    """主函数：使用DDP处理视频"""
    # 检查GPU数量
    if not torch.cuda.is_available():
        print("错误：未检测到CUDA设备。")
        return
    
    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 块可用的GPU，将使用DDP进行处理。")
    
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
        args=(world_size, input_dir, output_dir),
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
    
    # video_directory_list=["vggsound_04_3s"]
    # video_directory_list = ["vggsound_00_3s","vggsound_01_3s","vggsound_02_3s","vggsound_03_3s","vggsound_04_3s", 
    # "vggsound_06_3s", "vggsound_07_3s", "vggsound_08_3s", "vggsound_09_3s",]
    # video_directory_list = ["vggsound_10_3s", "vggsound_11_3s", "vggsound_12_3s", "vggsound_13_3s", "vggsound_14_3s",]
    video_directory_list =["vggsound_15_3s", "vggsound_16_3s", "vggsound_17_3s", "vggsound_18_3s", "vggsound_19_3s",]
    #     "vggsound_15_3s", "vggsound_16_3s", "vggsound_17_3s", "vggsound_18_3s", "vggsound_19_3s",
    # ]
    input_video_directory_base="/blob/vggsound_cropped"
    output_latent_directory_base = "/blob/vggsound_cropped_audio_latent_fixed"
   
    for video_dir in video_directory_list:
        print(video_dir)
        input_video_directory = os.path.join(input_video_directory_base, video_dir)
        output_latent_directory = os.path.join(output_latent_directory_base, video_dir)
        
        try:
            batch_process_videos_ddp(input_video_directory, output_latent_directory)
        except Exception as e:
            print(f"\n处理 {video_dir} 时发生严重错误: {e}")
            print(traceback.format_exc())
