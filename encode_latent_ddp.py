#!/usr/bin/env python
"""
独立GPU进程版本 - 每个GPU独立运行，无同步开销
"""

import os
import sys
import torch
import numpy as np
import torchaudio
import traceback
from pathlib import Path
from multiprocessing import Process
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import time
import json

# ============ 音频处理函数 ============
def get_mel_from_wav_batch(audio_batch, _stft):
    """处理一批音频"""
    if audio_batch.dim() == 3:
        audio_batch = audio_batch.squeeze(1)
    elif audio_batch.dim() == 1:
        audio_batch = audio_batch.unsqueeze(0)
    
    assert audio_batch.dim() == 2, f"Expected 2D tensor, got {audio_batch.dim()}D"
    
    audio_batch = torch.clip(audio_batch, -1, 1)
    audio_batch = torch.autograd.Variable(audio_batch, requires_grad=False)
    
    melspec, magnitudes, phases, energy = _stft.mel_spectrogram(audio_batch)
    return melspec, magnitudes, energy

def pad_wav(waveform, segment_length):
    """填充波形到指定长度"""
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length
    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    elif waveform_length < segment_length:
        temp_wav = np.zeros(segment_length)
        temp_wav[:waveform_length] = waveform
        return temp_wav

def _pad_spec(fbank, target_length=1024):
    """填充频谱到目标长度"""
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
    """归一化波形"""
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5

def read_wav_file(filename, segment_length):
    """读取并预处理音频文件"""
    try:
        waveform, sr = torchaudio.load(filename, format="mp4", backend="ffmpeg")
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        waveform = waveform.numpy()[0, ...]
        waveform = normalize_wav(waveform)
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
    
    for i, filename in enumerate(filenames):
        waveform = read_wav_file(filename, target_length * 160)
        if waveform is not None:
            batch_waveforms.append(waveform)
            valid_indices.append(i)
    
    if not batch_waveforms:
        return None, None, None, []
    
    batch_waveforms = torch.FloatTensor(np.stack(batch_waveforms)).to(device)
    fbank_batch, log_magnitudes_stft_batch, energy_batch = get_mel_from_wav_batch(
        batch_waveforms, fn_STFT
    )
    
    processed_fbanks = []
    processed_log_mags = []
    
    for i in range(fbank_batch.size(0)):
        fbank = fbank_batch[i].T
        log_mag = log_magnitudes_stft_batch[i].T
        
        fbank = _pad_spec(fbank, target_length)
        log_mag = _pad_spec(log_mag, target_length)
        
        processed_fbanks.append(fbank)
        processed_log_mags.append(log_mag)
    
    fbank_batch = torch.stack(processed_fbanks)
    log_magnitudes_stft_batch = torch.stack(processed_log_mags)
    
    return fbank_batch, log_magnitudes_stft_batch, batch_waveforms, valid_indices

def encode_audio_batch_from_videos(video_paths, vae, fn_STFT, device, vae_chunk_size=8):
    """批量编码多个视频的音频"""
    try:
        from audioldm2.utils import default_audioldm_config
        config = default_audioldm_config()
        duration = 3
        
        mel_batch, _, _, valid_indices = wav_to_fbank_batch(
            video_paths, 
            target_length=int(duration * 102.4), 
            fn_STFT=fn_STFT, 
            device=device
        )
        
        if mel_batch is None:
            return [None] * len(video_paths), ["FAILED"] * len(video_paths)
        
        mel_batch = mel_batch.unsqueeze(1).to(torch.float16)
        
        with torch.no_grad():
            if mel_batch.size(0) > vae_chunk_size:
                latent_list = []
                for i in range(0, mel_batch.size(0), vae_chunk_size):
                    batch_chunk = mel_batch[i:i+vae_chunk_size]
                    latent_chunk = vae.encode(batch_chunk).latent_dist.mode()
                    latent_list.append(latent_chunk)
                latent_representations = torch.cat(latent_list, dim=0)
            else:
                latent_representations = vae.encode(mel_batch).latent_dist.mode()
        
        results = [None] * len(video_paths)
        statuses = ["FAILED"] * len(video_paths)
        
        for i, valid_idx in enumerate(valid_indices):
            results[valid_idx] = latent_representations[i].cpu().numpy()
            statuses[valid_idx] = "SUCCESS"
        
        return results, statuses
        
    except Exception as e:
        print(f"批处理错误: {e}\n{traceback.format_exc()}")
        return [None] * len(video_paths), ["FAILED"] * len(video_paths)

# ============ 单GPU处理进程 ============
def single_gpu_worker(gpu_id: int, input_dirs: List[str], output_base: str, config: Dict):
    """单个GPU的独立处理进程"""
    # 设置环境变量，只使用指定的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 异步执行
    
    # 现在这个进程只能看到一个GPU
    device = torch.device("cuda:0")
    
    print(f"\n[GPU {gpu_id}] 启动进程")
    print(f"[GPU {gpu_id}] 负责处理 {len(input_dirs)} 个目录")
    
    try:
        # 导入必要的库（在设置CUDA_VISIBLE_DEVICES后）
        from diffusers import AutoencoderKL
        from audioldm2.utils import default_audioldm_config
        from audioldm2.utilities.audio.stft import TacotronSTFT
        
        # 加载VAE模型
        print(f"[GPU {gpu_id}] 加载VAE模型...")
        vae = AutoencoderKL.from_pretrained(
            "cvssp/audioldm2", 
            subfolder="vae", 
            torch_dtype=torch.float16,
            resume_download=True
        )
        vae = vae.to(device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        
        # 设置STFT
        config_audio = default_audioldm_config()
        fn_STFT = TacotronSTFT(
            config_audio["preprocessing"]["stft"]["filter_length"],
            config_audio["preprocessing"]["stft"]["hop_length"],
            config_audio["preprocessing"]["stft"]["win_length"],
            config_audio["preprocessing"]["mel"]["n_mel_channels"],
            config_audio["preprocessing"]["audio"]["sampling_rate"],
            config_audio["preprocessing"]["mel"]["mel_fmin"],
            config_audio["preprocessing"]["mel"]["mel_fmax"],
        ).to(device)
        
        # 获取GPU内存并选择batch size
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU {gpu_id}] 显存: {memory_gb:.1f}GB")
        
        if memory_gb >= 70:  # 80GB
            batch_size = config.get('batch_size', 32)
            vae_chunk_size = config.get('vae_chunk_size', 32)
        elif memory_gb >= 35:  # 40GB
            batch_size = config.get('batch_size', 32)
            vae_chunk_size = config.get('vae_chunk_size', 32)
        else:  # 24GB或更小
            batch_size = config.get('batch_size', 16)
            vae_chunk_size = config.get('vae_chunk_size', 16)
        
        print(f"[GPU {gpu_id}] 使用 batch_size={batch_size}, vae_chunk={vae_chunk_size}")
        
        # 统计信息
        total_processed = 0
        total_failed = 0
        total_skipped = 0
        start_time = time.time()
        
        # 处理每个分配的目录
        for dir_idx, input_dir in enumerate(input_dirs):
            dir_name = os.path.basename(input_dir)
            output_dir = os.path.join(output_base, dir_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取所有视频文件
            video_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mp4')])
            if not video_files:
                print(f"[GPU {gpu_id}] 目录 {dir_name} 没有视频文件")
                continue
            
            print(f"[GPU {gpu_id}] [{dir_idx+1}/{len(input_dirs)}] 处理 {dir_name}: {len(video_files)} 个文件")
            
            # 创建进度条（使用position参数避免重叠）
            pbar = tqdm(
                total=len(video_files),
                desc=f"GPU{gpu_id}-{dir_name}",
                position=gpu_id,
                leave=True
            )
            
            # 批量处理
            for i in range(0, len(video_files), batch_size):
                batch_files = video_files[i:i+batch_size]
                batch_paths = [os.path.join(input_dir, f) for f in batch_files]
                batch_outputs = [os.path.join(output_dir, f.replace('.mp4', '.npy')) for f in batch_files]
                
                # 检查哪些需要处理
                to_process = []
                to_save = []
                for path, output in zip(batch_paths, batch_outputs):
                    if not os.path.exists(output):
                        to_process.append(path)
                        to_save.append(output)
                    else:
                        total_skipped += 1
                
                if to_process:
                    # 批量编码
                    latents, statuses = encode_audio_batch_from_videos(
                        to_process, vae, fn_STFT, device, vae_chunk_size
                    )
                    
                    # 保存结果
                    for latent, status, output_path in zip(latents, statuses, to_save):
                        if status == "SUCCESS" and latent is not None:
                            np.save(output_path, latent)
                            total_processed += 1
                        else:
                            total_failed += 1
                            print(f"[GPU {gpu_id}] 失败: {os.path.basename(output_path)}")
                
                pbar.update(len(batch_files))
                
                # 定期清理缓存
                if (i // batch_size) % 10 == 0:
                    torch.cuda.empty_cache()
            
            pbar.close()
        
        # 统计信息
        elapsed_time = time.time() - start_time
        print(f"\n[GPU {gpu_id}] 完成!")
        print(f"  - 成功处理: {total_processed} 个文件")
        print(f"  - 跳过已存在: {total_skipped} 个文件")
        print(f"  - 处理失败: {total_failed} 个文件")
        print(f"  - 总用时: {elapsed_time/60:.1f} 分钟")
        print(f"  - 处理速度: {total_processed/elapsed_time:.2f} 文件/秒")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] 发生错误: {e}")
        traceback.print_exc()
    finally:
        # 清理
        if 'vae' in locals():
            del vae
        torch.cuda.empty_cache()

# ============ 主控制器 ============
class IndependentGPUProcessor:
    """独立GPU进程管理器"""
    
    def __init__(self, num_gpus: int = None):
        """初始化
        
        Args:
            num_gpus: 要使用的GPU数量，None表示使用所有可用GPU
        """
        available_gpus = torch.cuda.device_count()
        if num_gpus is None:
            self.num_gpus = available_gpus
        else:
            self.num_gpus = min(num_gpus, available_gpus)
        
        print(f"独立GPU处理器初始化")
        print(f"  - 可用GPU: {available_gpus}")
        print(f"  - 使用GPU: {self.num_gpus}")
    
    def distribute_directories(self, directories: List[str]) -> Dict[int, List[str]]:
        """将目录分配给各个GPU
        
        使用轮询方式均匀分配
        """
        assignments = {i: [] for i in range(self.num_gpus)}
        
        for idx, directory in enumerate(directories):
            gpu_id = idx % self.num_gpus
            assignments[gpu_id].append(directory)
        
        return assignments
    
    def process(self, input_base: str, output_base: str, dir_list: List[str], config: Dict = None):
        """启动独立GPU进程处理
        
        Args:
            input_base: 输入基础目录
            output_base: 输出基础目录
            dir_list: 要处理的子目录列表
            config: 配置参数（batch_size, vae_chunk_size等）
        """
        if config is None:
            config = {}
        
        # 准备完整路径
        full_paths = [os.path.join(input_base, d) for d in dir_list]
        
        # 分配工作
        gpu_assignments = self.distribute_directories(full_paths)
        
        # 显示分配情况
        print("\n工作分配:")
        for gpu_id, dirs in gpu_assignments.items():
            if dirs:
                print(f"  GPU {gpu_id}: {len(dirs)} 个目录 - {[os.path.basename(d) for d in dirs]}")
        
        # 启动进程
        processes = []
        print("\n启动GPU进程...")
        
        for gpu_id, assigned_dirs in gpu_assignments.items():
            if not assigned_dirs:
                continue
            
            # 创建进程
            p = Process(
                target=single_gpu_worker,
                args=(gpu_id, assigned_dirs, output_base, config)
            )
            p.start()
            processes.append((gpu_id, p))
            
            # 稍微错开启动，避免同时加载模型造成的内存峰值
            time.sleep(2)
        
        print(f"已启动 {len(processes)} 个GPU进程\n")
        
        # 等待所有进程完成
        for gpu_id, p in processes:
            p.join()
            print(f"GPU {gpu_id} 进程已完成")
        
        print("\n✅ 所有GPU进程已完成!")

# ============ 便捷函数 ============
def auto_process_videos(input_base: str, output_base: str, dir_list: List[str], 
                       num_gpus: int = None, batch_size: int = None, 
                       vae_chunk_size: int = None):
    """自动选择最优策略处理视频
    
    Args:
        input_base: 输入基础目录
        output_base: 输出基础目录  
        dir_list: 要处理的子目录列表
        num_gpus: 使用的GPU数量（None=全部）
        batch_size: 批处理大小（None=自动）
        vae_chunk_size: VAE分块大小（None=自动）
    """
    config = {}
    if batch_size is not None:
        config['batch_size'] = batch_size
    if vae_chunk_size is not None:
        config['vae_chunk_size'] = vae_chunk_size
    
    processor = IndependentGPUProcessor(num_gpus)
    processor.process(input_base, output_base, dir_list, config)

# ============ 主程序 ============
if __name__ == '__main__':
    import multiprocessing as mp
    
    # 设置启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # 配置
    # video_directory_list = [
    #     "vggsound_15_3s", 
    #     "vggsound_16_3s", 
    #     "vggsound_17_3s", 
    #     "vggsound_18_3s", 
    #     "vggsound_19_3s"
    # ]
    video_directory_list = [
        # "vggsound_00_3s", 
        # "vggsound_01_3s", 
        # "vggsound_02_3s", 
        # "vggsound_03_3s", 
        # "vggsound_04_3s",
        # "vggsound_05_3s", 
        # "vggsound_06_3s", 
        # "vggsound_07_3s", 
        # "vggsound_08_3s", 
        # "vggsound_09_3s",
        "vggsound_10_3s", 
        "vggsound_11_3s", 
        "vggsound_12_3s", 
        "vggsound_13_3s", 
        "vggsound_14_3s",
    ]
    
    input_video_directory_base = "/blob/vggsound_cropped"
    output_latent_directory_base = "/blob/vggsound_cropped_audio_latent_fixed"
    
    # 方式1: 完全自动（推荐）
    auto_process_videos(
        input_video_directory_base,
        output_latent_directory_base,
        video_directory_list
    )
    
    # 方式2: 指定GPU数量
    # auto_process_videos(
    #     input_video_directory_base,
    #     output_latent_directory_base,
    #     video_directory_list,
    #     num_gpus=4  # 只使用4个GPU
    # )
    
    # 方式3: 自定义配置
    # auto_process_videos(
    #     input_video_directory_base,
    #     output_latent_directory_base,
    #     video_directory_list,
    #     batch_size=32,      # 手动设置batch size
    #     vae_chunk_size=32   # 手动设置VAE chunk size
    # )
