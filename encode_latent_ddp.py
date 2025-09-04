#!/usr/bin/env python
"""
独立GPU进程版本 - 使用DataLoader优化I/O性能
"""

import os
import sys
import torch
import numpy as np
import torchaudio
import traceback
from pathlib import Path
from multiprocessing import Process
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import time
import json

# ============ 自定义Dataset ============
class VideoAudioDataset(Dataset):
    """视频音频数据集，支持高效批处理"""
    
    def __init__(self, video_paths: List[str], output_paths: List[str], 
                 segment_length: int = 16000 * 3, skip_existing: bool = True):
        """
        Args:
            video_paths: 视频文件路径列表
            output_paths: 对应的输出路径列表
            segment_length: 音频段长度
            skip_existing: 是否跳过已存在的输出文件
        """
        self.segment_length = segment_length
        
        # 过滤已存在的文件
        self.video_paths = []
        self.output_paths = []
        
        for video_path, output_path in zip(video_paths, output_paths):
            if skip_existing and os.path.exists(output_path):
                # 验证文件有效性
                try:
                    data = np.load(output_path)
                    if data.size > 0:
                        continue  # 跳过有效的已存在文件
                except:
                    pass  # 文件损坏，需要重新处理
            
            self.video_paths.append(video_path)
            self.output_paths.append(output_path)
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """加载单个样本"""
        video_path = self.video_paths[idx]
        output_path = self.output_paths[idx]
        
        try:
            # 读取音频
            waveform, sr = torchaudio.load(video_path, format="mp4", backend="ffmpeg")
            
            # 重采样到16kHz
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            waveform = waveform.squeeze(0)
            
            # 归一化
            waveform = waveform - torch.mean(waveform)
            max_val = torch.max(torch.abs(waveform))
            if max_val > 1e-8:
                waveform = waveform / max_val * 0.5
            
            # 填充或裁剪
            current_len = waveform.shape[0]
            if current_len > self.segment_length:
                waveform = waveform[:self.segment_length]
            elif current_len < self.segment_length:
                waveform = torch.nn.functional.pad(waveform, (0, self.segment_length - current_len))
            
            return {
                'waveform': waveform,
                'output_path': output_path,
                'status': 'success'
            }
        except Exception as e:
            # 返回错误标记
            return {
                'waveform': torch.zeros(self.segment_length),
                'output_path': output_path,
                'status': f'error: {str(e)}'
            }

def collate_fn(batch):
    """自定义批处理函数"""
    waveforms = []
    output_paths = []
    valid_indices = []
    
    for i, item in enumerate(batch):
        if item['status'] == 'success':
            waveforms.append(item['waveform'])
            output_paths.append(item['output_path'])
            valid_indices.append(i)
    
    if waveforms:
        waveforms = torch.stack(waveforms)
    else:
        waveforms = None
    
    return {
        'waveforms': waveforms,
        'output_paths': output_paths,
        'valid_indices': valid_indices
    }

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

# ============ 单GPU处理进程（DataLoader版本）============
def single_gpu_worker_dataloader(gpu_id: int, input_dirs: List[str], output_base: str, config: Dict):
    """使用DataLoader的单GPU处理进程"""
    
    # 设置环境变量，只使用指定的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    device = torch.device("cuda:0")
    
    print(f"\n[GPU {gpu_id}] 启动DataLoader优化进程")
    print(f"[GPU {gpu_id}] 负责处理 {len(input_dirs)} 个目录")
    
    try:
        # 导入必要的库
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
        
        # 获取GPU内存并选择配置
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU {gpu_id}] 显存: {memory_gb:.1f}GB")
        
        # DataLoader配置
        if memory_gb >= 70:  # 80GB
            batch_size = config.get('batch_size', 32)
            vae_chunk_size = config.get('vae_chunk_size', 32)
            num_workers = config.get('num_workers', 8)
            prefetch_factor = config.get('prefetch_factor', 4)
        elif memory_gb >= 35:  # 40GB
            batch_size = config.get('batch_size', 24)
            vae_chunk_size = config.get('vae_chunk_size', 24)
            num_workers = config.get('num_workers', 6)
            prefetch_factor = config.get('prefetch_factor', 3)
        else:  # 24GB或更小
            batch_size = config.get('batch_size', 16)
            vae_chunk_size = config.get('vae_chunk_size', 16)
            num_workers = config.get('num_workers', 4)
            prefetch_factor = config.get('prefetch_factor', 2)
        
        print(f"[GPU {gpu_id}] 配置: batch_size={batch_size}, vae_chunk={vae_chunk_size}, workers={num_workers}")
        
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
            
            # 准备路径
            video_paths = [os.path.join(input_dir, f) for f in video_files]
            output_paths = [os.path.join(output_dir, f.replace('.mp4', '.npy')) for f in video_files]
            
            # 统计已存在的文件
            existing_count = sum(1 for p in output_paths if os.path.exists(p))
            total_skipped += existing_count
            
            # 创建数据集
            dataset = VideoAudioDataset(
                video_paths, 
                output_paths, 
                segment_length=16000 * 3,
                skip_existing=True
            )
            
            if len(dataset) == 0:
                print(f"[GPU {gpu_id}] [{dir_idx+1}/{len(input_dirs)}] {dir_name}: 所有 {len(video_files)} 个文件已处理")
                continue
            
            print(f"[GPU {gpu_id}] [{dir_idx+1}/{len(input_dirs)}] {dir_name}: 处理 {len(dataset)}/{len(video_files)} 个文件")
            
            # 创建DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=prefetch_factor,
                persistent_workers=True if num_workers > 0 else False,
                collate_fn=collate_fn,
                drop_last=False
            )
            
            # 创建进度条
            pbar = tqdm(
                total=len(dataset),
                desc=f"GPU{gpu_id}-{dir_name}",
                position=gpu_id,
                leave=True
            )
            
            # 处理批次
            for batch_idx, batch in enumerate(dataloader):
                if batch['waveforms'] is None or len(batch['waveforms']) == 0:
                    continue
                
                try:
                    # 移动到GPU
                    waveforms = batch['waveforms'].to(device, non_blocking=True)
                    output_paths_batch = batch['output_paths']
                    
                    # 计算mel频谱
                    with torch.no_grad():
                        # 处理音频到mel
                        mel_batch, _, _ = get_mel_from_wav_batch(waveforms, fn_STFT)
                        
                        # 处理mel频谱维度
                        processed_mels = []
                        target_length = int(3 * 102.4)  # duration * 102.4
                        
                        for i in range(mel_batch.size(0)):
                            mel = mel_batch[i].T
                            mel = _pad_spec(mel, target_length)
                            processed_mels.append(mel)
                        
                        mel_batch = torch.stack(processed_mels)
                        mel_batch = mel_batch.unsqueeze(1).to(torch.float16)
                        
                        # VAE编码（分块处理以节省内存）
                        if mel_batch.size(0) > vae_chunk_size:
                            latent_list = []
                            for i in range(0, mel_batch.size(0), vae_chunk_size):
                                chunk = mel_batch[i:i+vae_chunk_size]
                                latent_chunk = vae.encode(chunk).latent_dist.mode()
                                latent_list.append(latent_chunk.cpu())
                            latents = torch.cat(latent_list, dim=0).numpy()
                        else:
                            latents = vae.encode(mel_batch).latent_dist.mode().cpu().numpy()
                    
                    # 保存结果
                    for latent, output_path in zip(latents, output_paths_batch):
                        try:
                            # 原子写入避免损坏
                            temp_path = f"{output_path}.tmp_{gpu_id}"
                            np.save(temp_path, latent)
                            os.rename(temp_path, output_path)
                            total_processed += 1
                        except Exception as e:
                            print(f"[GPU {gpu_id}] 保存失败 {os.path.basename(output_path)}: {e}")
                            total_failed += 1
                    
                except Exception as e:
                    print(f"[GPU {gpu_id}] 批处理错误: {e}")
                    total_failed += len(batch['output_paths'])
                
                pbar.update(len(batch['output_paths']))
                
                # 定期清理缓存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            pbar.close()
        
        # 统计信息
        elapsed_time = time.time() - start_time
        print(f"\n[GPU {gpu_id}] 完成!")
        print(f"  - 成功处理: {total_processed} 个文件")
        print(f"  - 跳过已存在: {total_skipped} 个文件")
        print(f"  - 处理失败: {total_failed} 个文件")
        print(f"  - 总用时: {elapsed_time/60:.1f} 分钟")
        if total_processed > 0:
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
class DataLoaderGPUProcessor:
    """DataLoader优化的GPU进程管理器"""
    
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
        
        print(f"DataLoader GPU处理器初始化")
        print(f"  - 可用GPU: {available_gpus}")
        print(f"  - 使用GPU: {self.num_gpus}")
    
    def distribute_directories(self, directories: List[str]) -> Dict[int, List[str]]:
        """将目录分配给各个GPU"""
        assignments = {i: [] for i in range(self.num_gpus)}
        
        for idx, directory in enumerate(directories):
            gpu_id = idx % self.num_gpus
            assignments[gpu_id].append(directory)
        
        return assignments
    
    def process(self, input_base: str, output_base: str, dir_list: List[str], config: Dict = None):
        """启动DataLoader优化的GPU进程处理"""
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
        print("\n启动GPU进程（DataLoader优化）...")
        
        for gpu_id, assigned_dirs in gpu_assignments.items():
            if not assigned_dirs:
                continue
            
            # 创建进程
            p = Process(
                target=single_gpu_worker_dataloader,
                args=(gpu_id, assigned_dirs, output_base, config)
            )
            p.start()
            processes.append((gpu_id, p))
            
            # 稍微错开启动
            time.sleep(2)
        
        print(f"已启动 {len(processes)} 个GPU进程\n")
        
        # 等待所有进程完成
        for gpu_id, p in processes:
            p.join()
            print(f"GPU {gpu_id} 进程已完成")
        
        print("\n✅ 所有GPU进程已完成!")

# ============ 进度检查函数 ============
def check_progress(input_base: str, output_base: str, dir_list: List[str]):
    """检查处理进度"""
    print("\n📊 处理进度统计:")
    print("-" * 60)
    
    total_videos = 0
    total_processed = 0
    total_corrupted = 0
    
    for dir_name in dir_list:
        input_dir = os.path.join(input_base, dir_name)
        output_dir = os.path.join(output_base, dir_name)
        
        if not os.path.exists(input_dir):
            print(f"❌ {dir_name}: 输入目录不存在")
            continue
        
        # 统计视频文件
        video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
        num_videos = len(video_files)
        total_videos += num_videos
        
        # 统计已处理的文件
        if os.path.exists(output_dir):
            processed = 0
            corrupted = 0
            
            for video_file in video_files:
                output_file = os.path.join(output_dir, video_file.replace('.mp4', '.npy'))
                if os.path.exists(output_file):
                    # 检查文件是否有效
                    try:
                        data = np.load(output_file)
                        if data.size > 0:
                            processed += 1
                        else:
                            corrupted += 1
                    except:
                        corrupted += 1
            
            total_processed += processed
            total_corrupted += corrupted
            
            percentage = (processed / num_videos * 100) if num_videos > 0 else 0
            status = "✅" if processed == num_videos else "🔄"
            
            print(f"{status} {dir_name}: {processed}/{num_videos} ({percentage:.1f}%)")
            if corrupted > 0:
                print(f"   ⚠️ 损坏文件: {corrupted}")
        else:
            print(f"⏳ {dir_name}: 0/{num_videos} (0.0%) - 输出目录不存在")
    
    print("-" * 60)
    overall_percentage = (total_processed / total_videos * 100) if total_videos > 0 else 0
    print(f"📈 总体进度: {total_processed}/{total_videos} ({overall_percentage:.1f}%)")
    if total_corrupted > 0:
        print(f"⚠️ 总损坏文件: {total_corrupted}")
    
    return total_processed, total_videos

# ============ 便捷函数 ============
def auto_process_videos_dataloader(input_base: str, output_base: str, dir_list: List[str], 
                                  num_gpus: int = None, batch_size: int = None, 
                                  num_workers: int = None):
    """使用DataLoader优化的自动处理函数
    
    Args:
        input_base: 输入基础目录
        output_base: 输出基础目录  
        dir_list: 要处理的子目录列表
        num_gpus: 使用的GPU数量（None=全部）
        batch_size: 批处理大小（None=自动）
        num_workers: DataLoader工作线程数（None=自动）
    """
    config = {}
    if batch_size is not None:
        config['batch_size'] = batch_size
    if num_workers is not None:
        config['num_workers'] = num_workers
    
    processor = DataLoaderGPUProcessor(num_gpus)
    processor.process(input_base, output_base, dir_list, config)

# ============ 主程序 ============
if __name__ == '__main__':
    import multiprocessing as mp
    import argparse
    
    # 参数解析
    parser = argparse.ArgumentParser(description='批量处理视频音频编码（DataLoader优化版）')
    parser.add_argument('--check', action='store_true', help='只检查进度，不处理')
    parser.add_argument('--gpus', type=int, help='使用的GPU数量')
    parser.add_argument('--batch-size', type=int, help='批处理大小')
    parser.add_argument('--workers', type=int, help='DataLoader工作线程数')
    
    args = parser.parse_args()
    
    # 设置启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # 配置
    video_directory_list = [
        "vggsound_00_3s", 
        "vggsound_01_3s", 
        "vggsound_02_3s", 
        "vggsound_03_3s", 
        "vggsound_04_3s",
        "vggsound_05_3s", 
        "vggsound_06_3s", 
        "vggsound_07_3s", 
        "vggsound_08_3s", 
        "vggsound_09_3s",
    ]
    
    input_video_directory_base = "/blob/vggsound_cropped"
    output_latent_directory_base = "/blob/vggsound_cropped_audio_latent_fixed"
    
    # 检查进度
    if args.check:
        check_progress(
            input_video_directory_base,
            output_latent_directory_base,
            video_directory_list
        )
    else:
        # 运行处理
        auto_process_videos_dataloader(
            input_video_directory_base,
            output_latent_directory_base,
            video_directory_list,
            num_gpus=args.gpus,
            batch_size=args.batch_size,
            num_workers=args.workers
        )
