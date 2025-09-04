import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchaudio
from pathlib import Path
import time
from typing import List, Tuple, Optional
import os
from tqdm import tqdm
import traceback

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
        
        print(f"数据集: {len(self.video_paths)} 个文件需要处理")
    
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
            
            # 归一化和填充
            waveform = waveform.squeeze(0)
            waveform = self.normalize_wav(waveform)
            waveform = self.pad_wav(waveform)
            
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
    
    def normalize_wav(self, waveform):
        """归一化波形"""
        waveform = waveform - torch.mean(waveform)
        max_val = torch.max(torch.abs(waveform))
        if max_val > 1e-8:
            waveform = waveform / max_val * 0.5
        return waveform
    
    def pad_wav(self, waveform):
        """填充或裁剪波形"""
        current_len = waveform.shape[0]
        if current_len == self.segment_length:
            return waveform
        elif current_len > self.segment_length:
            return waveform[:self.segment_length]
        else:
            return torch.nn.functional.pad(waveform, (0, self.segment_length - current_len))

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

# ============ 优化的处理函数 ============
def process_with_dataloader(gpu_id: int, input_dirs: List[str], output_base: str, config: Dict):
    """使用DataLoader的优化处理函数"""
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0")
    
    print(f"\n[GPU {gpu_id}] 启动DataLoader版本")
    
    try:
        # 导入模型
        from diffusers import AutoencoderKL
        from audioldm2.utils import default_audioldm_config
        from audioldm2.utilities.audio.stft import TacotronSTFT
        
        # 加载模型
        vae = AutoencoderKL.from_pretrained(
            "cvssp/audioldm2", 
            subfolder="vae", 
            torch_dtype=torch.float16
        ).to(device).eval()
        
        # STFT
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
        
        # 获取配置
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # DataLoader配置
        if memory_gb >= 70:  # 80GB GPU
            batch_size = config.get('batch_size', 32)
            num_workers = config.get('num_workers', 8)
            prefetch_factor = config.get('prefetch_factor', 4)
        elif memory_gb >= 35:  # 40GB GPU
            batch_size = config.get('batch_size', 24)
            num_workers = config.get('num_workers', 6)
            prefetch_factor = config.get('prefetch_factor', 3)
        else:  # 24GB GPU
            batch_size = config.get('batch_size', 16)
            num_workers = config.get('num_workers', 4)
            prefetch_factor = config.get('prefetch_factor', 2)
        
        print(f"[GPU {gpu_id}] 配置: batch_size={batch_size}, workers={num_workers}")
        
        total_processed = 0
        total_failed = 0
        
        # 处理每个目录
        for dir_idx, input_dir in enumerate(input_dirs):
            dir_name = os.path.basename(input_dir)
            output_dir = os.path.join(output_base, dir_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # 准备文件列表
            video_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mp4')])
            video_paths = [os.path.join(input_dir, f) for f in video_files]
            output_paths = [os.path.join(output_dir, f.replace('.mp4', '.npy')) for f in video_files]
            
            # 创建数据集
            dataset = VideoAudioDataset(video_paths, output_paths, skip_existing=True)
            
            if len(dataset) == 0:
                print(f"[GPU {gpu_id}] {dir_name}: 所有文件已处理")
                continue
            
            # 创建DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=prefetch_factor,
                persistent_workers=True if num_workers > 0 else False,
                collate_fn=collate_fn
            )
            
            # 处理批次
            pbar = tqdm(
                total=len(dataset),
                desc=f"GPU{gpu_id}-{dir_name}",
                position=gpu_id
            )
            
            for batch_idx, batch in enumerate(dataloader):
                if batch['waveforms'] is None:
                    continue
                
                # 移动到GPU
                waveforms = batch['waveforms'].to(device, non_blocking=True)
                output_paths_batch = batch['output_paths']
                
                # 计算mel频谱
                with torch.no_grad():
                    # 处理音频到mel
                    mel_batch, _, _ = get_mel_from_wav_batch(waveforms, fn_STFT)
                    mel_batch = mel_batch.unsqueeze(1).to(torch.float16)
                    
                    # VAE编码
                    latents = vae.encode(mel_batch).latent_dist.mode()
                    latents = latents.cpu().numpy()
                
                # 保存结果
                for latent, output_path in zip(latents, output_paths_batch):
                    try:
                        # 原子写入
                        temp_path = f"{output_path}.tmp_{gpu_id}"
                        np.save(temp_path, latent)
                        os.rename(temp_path, output_path)
                        total_processed += 1
                    except Exception as e:
                        print(f"[GPU {gpu_id}] 保存失败: {e}")
                        total_failed += 1
                
                pbar.update(len(output_paths_batch))
                
                # 定期清理缓存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            pbar.close()
        
        print(f"[GPU {gpu_id}] 完成: 成功={total_processed}, 失败={total_failed}")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] 错误: {e}")
        traceback.print_exc()

# ============ 性能基准测试 ============
class PerformanceBenchmark:
    """性能对比测试"""
    
    @staticmethod
    def benchmark_no_dataloader(video_paths: List[str], batch_size: int = 16):
        """测试无DataLoader版本"""
        start_time = time.time()
        processed = 0
        
        for i in range(0, len(video_paths), batch_size):
            batch = video_paths[i:i+batch_size]
            # 串行读取
            for path in batch:
                try:
                    waveform, sr = torchaudio.load(path, format="mp4", backend="ffmpeg")
                    processed += 1
                except:
                    pass
        
        elapsed = time.time() - start_time
        return processed, elapsed, processed / elapsed
    
    @staticmethod
    def benchmark_with_dataloader(video_paths: List[str], batch_size: int = 16, num_workers: int = 4):
        """测试DataLoader版本"""
        dataset = VideoAudioDataset(video_paths, [f"{p}.npy" for p in video_paths])
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2
        )
        
        start_time = time.time()
        processed = 0
        
        for batch in dataloader:
            processed += len(batch['valid_indices'])
        
        elapsed = time.time() - start_time
        return processed, elapsed, processed / elapsed

# ============ 性能对比脚本 ============
def compare_performance(input_dir: str, num_files: int = 100):
    """对比两种方法的性能"""
    print("\n📊 性能对比测试")
    print("=" * 60)
    
    # 获取测试文件
    video_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.mp4')][:num_files]
    
    if not video_files:
        print("没有找到视频文件")
        return
    
    print(f"测试文件数: {len(video_files)}")
    
    # 测试不同配置
    configs = [
        {"batch_size": 8, "num_workers": 0, "name": "单线程"},
        {"batch_size": 8, "num_workers": 2, "name": "2线程"},
        {"batch_size": 8, "num_workers": 4, "name": "4线程"},
        {"batch_size": 16, "num_workers": 4, "name": "大批次+4线程"},
    ]
    
    print("\n无DataLoader版本:")
    benchmark = PerformanceBenchmark()
    processed, elapsed, throughput = benchmark.benchmark_no_dataloader(video_files, batch_size=16)
    baseline_throughput = throughput
    print(f"  处理: {processed} 文件")
    print(f"  耗时: {elapsed:.2f} 秒")
    print(f"  吞吐: {throughput:.2f} 文件/秒")
    
    print("\nDataLoader版本:")
    for config in configs:
        processed, elapsed, throughput = benchmark.benchmark_with_dataloader(
            video_files, 
            batch_size=config["batch_size"],
            num_workers=config["num_workers"]
        )
        speedup = throughput / baseline_throughput
        print(f"\n  {config['name']}:")
        print(f"    处理: {processed} 文件")
        print(f"    耗时: {elapsed:.2f} 秒")
        print(f"    吞吐: {throughput:.2f} 文件/秒")
        print(f"    加速: {speedup:.2f}x")

# ============ 主程序 ============
if __name__ == '__main__':
    import argparse
    import multiprocessing as mp
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true', help='运行性能测试')
    parser.add_argument('--use-dataloader', action='store_true', help='使用DataLoader版本')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader工作线程数')
    
    args = parser.parse_args()
    
    if args.benchmark:
        # 运行性能对比
        compare_performance("/blob/vggsound_cropped/vggsound_15_3s", num_files=100)
    else:
        # 正常处理
        video_directory_list = [
            "vggsound_15_3s", 
            "vggsound_16_3s", 
            "vggsound_17_3s", 
            "vggsound_18_3s", 
            "vggsound_19_3s"
        ]
        
        config = {
            'num_workers': args.workers,
            'prefetch_factor': 2
        }
        
        if args.use_dataloader:
            print("使用DataLoader优化版本")
            # 使用DataLoader版本
            mp.set_start_method('spawn', force=True)
            # ... 启动process_with_dataloader
        else:
            print("使用原始版本")
            # 使用原始版本
