import torch
from diffusers import AudioLDM2Pipeline
import numpy as np
from scipy.io.wavfile import write
import os
import scipy
import argparse
import torchaudio
import subprocess
import tempfile
import torch.nn.functional as F
import sys

# ===================================================================
#               (新增) 原始的波形归一化函数
# ===================================================================
def normalize_wav(waveform):
    """
    移除直流分量，然后将波形归一化到 [-0.5, 0.5] 的范围内。
    这很可能是原始编码器使用的预处理步骤。
    """
    # waveform = waveform - np.mean(waveform)
    # # 加上一个极小值以防止除以零
    # waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    # return waveform * 0.5
    return waveform 

# ===================================================================
#               多分辨率声谱图损失
# ===================================================================
class MultiResolutionSpectrogramLoss:
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[160, 320, 640], 
                 win_lengths=[512, 1024, 2048], window='hann_window'):
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.window = window
        
    def stft(self, x, fft_size, hop_size, win_length):
        """Compute STFT magnitude"""
        window = getattr(torch, self.window)(win_length).to(x.device)
        stft = torch.stft(
            x, 
            n_fft=fft_size, 
            hop_length=hop_size, 
            win_length=win_length,
            window=window,
            return_complex=True
        )
        return torch.abs(stft)
    
    def compute_loss(self, pred, target):
        """Compute multi-resolution spectrogram loss"""
        pred = torch.from_numpy(pred).float() if isinstance(pred, np.ndarray) else pred
        target = torch.from_numpy(target).float() if isinstance(target, np.ndarray) else target
        
        if pred.dim() == 1: pred = pred.unsqueeze(0)
        if target.dim() == 1: target = target.unsqueeze(0)
            
        total_loss = 0.0
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            pred_spec = self.stft(pred, fft_size, hop_size, win_length)
            target_spec = self.stft(target, fft_size, hop_size, win_length)
            total_loss += F.l1_loss(pred_spec, target_spec) + F.mse_loss(torch.log(pred_spec + 1e-7), torch.log(target_spec + 1e-7))
            
        return total_loss / len(self.fft_sizes)

# --- 参数解析 ---
parser = argparse.ArgumentParser(description='从潜在表示还原音频并计算指标')
parser.add_argument('--input_latent_path', type=str, required=True, help='输入的 latent npy 文件路径')
parser.add_argument('--output_dir', type=str, default='/blob/avtok/', help='输出目录')
parser.add_argument('--device', type=str, default='auto', choices=['cuda', 'cpu', 'auto'], help='运行设备')
args = parser.parse_args()

origin_video_base_dir = "/blob/vggsound_cropped/"
latent_base_dir = "/blob/vggsound_cropped_audio_latent_fixed/"

# --- 1. 设置文件路径 ---
input_latent_path = args.input_latent_path
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(input_latent_path):
    raise FileNotFoundError(f"错误：找不到 latent 文件 '{input_latent_path}'。")

relative_path = input_latent_path.replace(latent_base_dir, "")
video_relative_path = relative_path.replace(".npy", ".mp4")
original_video_path = os.path.join(origin_video_base_dir, video_relative_path)
base_name = os.path.splitext(os.path.basename(input_latent_path))[0]

reconstructed_audio_path = os.path.join(output_dir, f"{base_name}_reconstructed.wav")
original_audio_path = os.path.join(output_dir, f"{base_name}_original.wav")

print(f"原始视频路径: {original_video_path}")
print(f"重建音频将保存至: {reconstructed_audio_path}")
print(f"原始音频将保存至: {original_audio_path}")

# --- 2. 从原始视频提取音频并重采样 ---
waveform_original = None
if os.path.exists(original_video_path):
    print(f"\n正在从视频提取并重采样原始音频...")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
    try:
        cmd = ['ffmpeg', '-i', original_video_path, '-vn', '-ar', '16000', '-ac', '1', '-f', 'wav', '-y', temp_audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            waveform_original, sr = torchaudio.load(temp_audio_path)
            waveform_original = waveform_original.squeeze().numpy()
            scipy.io.wavfile.write(original_audio_path, rate=16000, data=waveform_original)
            print(f"原始音频已提取并重采样，保存至: {original_audio_path}")
        else:
            print(f"警告：无法从视频提取音频。错误: {result.stderr}")
    finally:
        if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
else:
    print(f"警告：找不到原始视频文件 '{original_video_path}'")

# --- 3. 加载 AudioLDM 2 模型 ---
print("\n正在加载 AudioLDM 2 模型...")
repo_id = "cvssp/audioldm2"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
device = args.device if args.device != 'auto' else ("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipe.to(device)
print(f"模型已加载到 {device} 设备。")

# --- 4. 加载 Latent 并解码 ---
print(f"\n正在从 '{input_latent_path}' 加载潜在表示并解码...")
latent_np = np.load(input_latent_path)
if latent_np.ndim == 4: latent_np = latent_np.squeeze(0)
latent_tensor = torch.from_numpy(latent_np).to(device, dtype=torch.float16).unsqueeze(0)

with torch.no_grad():
    decoded_mel = pipe.vae.decode(latent_tensor).sample
    waveform = pipe.mel_spectrogram_to_waveform(decoded_mel).squeeze().cpu().numpy().astype(np.float32)


scipy.io.wavfile.write(reconstructed_audio_path, rate=16000, data=waveform)
print(f"重建音频已保存至: '{reconstructed_audio_path}'")

# --- 5. 计算损失指标 ---
if waveform_original is not None:
    print("\n--- 计算损失指标 ---")
    
    # ===================================================================
    #   (核心修改) 使用新发现的 normalize_wav 函数处理原始音频
    # ===================================================================
    print("\n正在对原始音频应用自定义峰值归一化 (peak=0.5)...")
    waveform_original_normalized = normalize_wav(waveform_original)
    waveform=normalize_wav(waveform)

    
    # (可选) 保存归一化后的原始音频，用于听感对比
    normalized_original_path = os.path.join(output_dir, f"{base_name}_original_normalized.wav")
    scipy.io.wavfile.write(normalized_original_path, rate=16000, data=waveform_original_normalized)
    print(f"归一化后的原始音频已保存至: {normalized_original_path}")
    # ===================================================================
    print(len(waveform), len(waveform_original_normalized))
    # 确保两个音频长度相同
    min_length = min(len(waveform), len(waveform_original_normalized),20000)
    waveform_recon_aligned = waveform[:min_length]
    # (重要) 使用归一化后的版本进行所有后续比较
    waveform_original_aligned = waveform_original_normalized[:min_length]
    
    # 1. 计算 L1 Loss (Waveform)
    l1_loss = np.mean(np.abs(waveform_recon_aligned - waveform_original_aligned))
    print(f"\nWaveform L1 Loss: {l1_loss:.6f}")
    
    # 2. 计算 L2 Loss (Waveform)
    l2_loss = np.mean((waveform_recon_aligned - waveform_original_aligned) ** 2)
    print(f"Waveform L2 Loss (MSE): {l2_loss:.6f}")
    
    # 3. 计算 Multi-Resolution Spectrogram Loss
    print("\n计算 Multi-Resolution Spectrogram Loss...")
    spec_loss_calculator = MultiResolutionSpectrogramLoss()
    spec_loss = spec_loss_calculator.compute_loss(waveform_recon_aligned, waveform_original_aligned)
    print(f"Multi-Resolution Spectrogram Loss: {spec_loss.item():.6f}")
    
    # 4. 计算信噪比 (SNR)
    signal_power = np.mean(waveform_original_aligned ** 2)
    noise_power = l2_loss
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    print(f"\nSignal-to-Noise Ratio (SNR): {snr:.2f} dB")
    
    # 5. 计算相关系数
    correlation = np.corrcoef(waveform_recon_aligned, waveform_original_aligned)[0, 1]
    print(f"Correlation Coefficient: {correlation:.4f}")
    
    # 6. 保存所有指标到文件
    metrics_path = os.path.join(output_dir, f"{base_name}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"音频重建质量评估指标\n" + "="*40 + "\n")
        f.write(f"文件: {base_name}\n")
        f.write(f"归一化方法: 自定义峰值归一化 (peak=0.5)\n\n")
        f.write(f"损失指标:\n" + "-"*20 + "\n")
        f.write(f"Waveform L1 Loss: {l1_loss:.6f}\n")
        f.write(f"Waveform L2 Loss (MSE): {l2_loss:.6f}\n")
        f.write(f"Multi-Resolution Spectrogram Loss: {spec_loss.item():.6f}\n\n")
        f.write(f"质量指标:\n" + "-"*20 + "\n")
        f.write(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB\n")
        f.write(f"Correlation Coefficient: {correlation:.4f}\n")
    
    print(f"\n指标已保存至: {metrics_path}")
else:
    print("\n警告：无法计算损失指标，因为原始音频不可用。")

print(f"\n所有文件都保存在: '{output_dir}'")