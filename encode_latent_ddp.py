import torch
import os
import subprocess
import multiprocessing as mp
from pathlib import Path
import json
import time
from typing import List, Dict
import argparse

# ============ 单GPU处理脚本 ============
def single_gpu_process(gpu_id: int, input_dirs: List[str], output_base: str, config: Dict):
    """单个GPU的处理函数"""
    # 设置当前进程只使用指定的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    import torch
    from diffusers import AutoencoderKL
    from audioldm2.utils import default_audioldm_config
    from audioldm2.utilities.audio.stft import TacotronSTFT
    import numpy as np
    from tqdm import tqdm
    import traceback
    
    device = torch.device("cuda:0")  # 因为CUDA_VISIBLE_DEVICES，这里始终是cuda:0
    
    print(f"[GPU {gpu_id}] 开始处理，负责目录: {input_dirs}")
    
    try:
        # 加载模型
        vae = AutoencoderKL.from_pretrained(
            "cvssp/audioldm2", 
            subfolder="vae", 
            torch_dtype=torch.float16,
            resume_download=True
        )
        vae = vae.to(device)
        vae.eval()
        
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
        
        # 根据GPU内存选择batch size
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if memory_gb >= 70:
            batch_size = config.get('batch_size_80gb', 32)
            vae_chunk_size = config.get('vae_chunk_size_80gb', 32)
        elif memory_gb >= 35:
            batch_size = config.get('batch_size_40gb', 24)
            vae_chunk_size = config.get('vae_chunk_size_40gb', 24)
        else:
            batch_size = config.get('batch_size_24gb', 16)
            vae_chunk_size = config.get('vae_chunk_size_24gb', 16)
        
        print(f"[GPU {gpu_id}] 内存: {memory_gb:.1f}GB, batch_size: {batch_size}")
        
        total_processed = 0
        total_failed = 0
        
        # 处理分配的目录
        for input_dir in input_dirs:
            output_dir = os.path.join(output_base, os.path.basename(input_dir))
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取所有视频文件
            video_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mp4')])
            
            # 创建进度条
            pbar = tqdm(
                total=len(video_files),
                desc=f"[GPU {gpu_id}] {os.path.basename(input_dir)}",
                position=gpu_id
            )
            
            # 批量处理
            for i in range(0, len(video_files), batch_size):
                batch_files = video_files[i:i+batch_size]
                batch_paths = [os.path.join(input_dir, f) for f in batch_files]
                batch_outputs = [os.path.join(output_dir, f.replace('.mp4', '.npy')) for f in batch_files]
                
                # 跳过已存在的文件
                to_process = []
                to_save = []
                for path, output in zip(batch_paths, batch_outputs):
                    if not os.path.exists(output):
                        to_process.append(path)
                        to_save.append(output)
                
                if to_process:
                    # 处理批次
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
                
                pbar.update(len(batch_files))
                
                # 定期清理缓存
                if i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
            
            pbar.close()
        
        print(f"[GPU {gpu_id}] 完成! 成功: {total_processed}, 失败: {total_failed}")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] 错误: {e}")
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()

# ============ 主控制脚本 ============
class MultiGPUProcessor:
    """多GPU独立进程管理器"""
    
    def __init__(self, num_gpus: int = None):
        if num_gpus is None:
            self.num_gpus = torch.cuda.device_count()
        else:
            self.num_gpus = min(num_gpus, torch.cuda.device_count())
        
        print(f"初始化 {self.num_gpus} 个GPU进程")
    
    def distribute_work(self, input_dirs: List[str]) -> Dict[int, List[str]]:
        """将工作负载分配给各个GPU"""
        gpu_assignments = {i: [] for i in range(self.num_gpus)}
        
        # 轮询分配
        for idx, dir_path in enumerate(input_dirs):
            gpu_id = idx % self.num_gpus
            gpu_assignments[gpu_id].append(dir_path)
        
        return gpu_assignments
    
    def process_directories(self, input_base: str, output_base: str, dir_list: List[str], config: Dict = None):
        """使用多个独立进程处理目录"""
        if config is None:
            config = {}
        
        # 准备完整路径
        input_dirs = [os.path.join(input_base, d) for d in dir_list]
        
        # 分配工作
        gpu_assignments = self.distribute_work(input_dirs)
        
        # 显示分配情况
        print("\n工作分配:")
        for gpu_id, dirs in gpu_assignments.items():
            print(f"  GPU {gpu_id}: {len(dirs)} 个目录")
        
        # 启动独立进程
        processes = []
        for gpu_id, assigned_dirs in gpu_assignments.items():
            if not assigned_dirs:
                continue
            
            # 创建独立进程
            p = mp.Process(
                target=single_gpu_process,
                args=(gpu_id, assigned_dirs, output_base, config)
            )
            p.start()
            processes.append(p)
            
            # 稍微错开启动时间，避免同时加载模型
            time.sleep(2)
        
        # 等待所有进程完成
        for p in processes:
            p.join()
        
        print("\n所有GPU进程已完成!")

# ============ 优化的批处理函数 ============
def process_with_optimal_strategy(input_base: str, output_base: str, dir_list: List[str]):
    """自动选择最优处理策略"""
    num_gpus = torch.cuda.device_count()
    num_dirs = len(dir_list)
    
    print(f"检测到 {num_gpus} 个GPU，需要处理 {num_dirs} 个目录")
    
    # 策略选择
    if num_gpus == 1:
        print("使用单GPU模式")
        # 单GPU直接处理
        single_gpu_process(0, [os.path.join(input_base, d) for d in dir_list], output_base, {})
    
    elif num_dirs >= num_gpus:
        print("使用多GPU独立进程模式（推荐）")
        # 多个目录，使用独立进程
        processor = MultiGPUProcessor(num_gpus)
        processor.process_directories(input_base, output_base, dir_list)
    
    else:
        print("使用GPU子集模式")
        # 目录少于GPU，只使用部分GPU
        processor = MultiGPUProcessor(num_dirs)
        processor.process_directories(input_base, output_base, dir_list)

# ============ 性能监控脚本 ============
def monitor_gpu_usage():
    """实时监控GPU使用情况"""
    import nvidia_ml_py3 as nvml
    
    nvml.nvmlInit()
    device_count = nvml.nvmlDeviceGetCount()
    
    print("GPU使用情况监控:")
    print("-" * 80)
    
    while True:
        for i in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            
            # 获取GPU信息
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            
            # 显示信息
            print(f"GPU {i}: 利用率={util.gpu}% | "
                  f"显存={mem_info.used/1024**3:.1f}/{mem_info.total/1024**3:.1f}GB | "
                  f"温度={temp}°C")
        
        print("-" * 80)
        time.sleep(2)

# ============ 主程序 ============
if __name__ == '__main__':
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 配置
    video_directory_list = [
        "vggsound_15_3s", 
        "vggsound_16_3s", 
        "vggsound_17_3s", 
        "vggsound_18_3s", 
        "vggsound_19_3s"
    ]
    
    input_video_directory_base = "/blob/vggsound_cropped"
    output_latent_directory_base = "/blob/vggsound_cropped_audio_latent_fixed"
    
    # 自定义配置（可选）
    custom_config = {
        'batch_size_24gb': 16,
        'vae_chunk_size_24gb': 16,
        'batch_size_40gb': 24,
        'vae_chunk_size_40gb': 24,
        'batch_size_80gb': 32,
        'vae_chunk_size_80gb': 32,
    }
    
    # 使用最优策略处理
    process_with_optimal_strategy(
        input_video_directory_base,
        output_latent_directory_base,
        video_directory_list
    )
