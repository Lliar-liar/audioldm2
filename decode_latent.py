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

# --- (重要) 确保可以导入 audioldm2 工具 ---
# 这一部分可能需要根据你的项目结构进行调整
# 假设 audioldm2 库安装在你的环境中，或者像你第一个脚本那样添加路径
try:
    from audioldm2.utils import default_audioldm_config
    from audioldm2.utilities.audio.stft import TacotronSTFT
except ImportError:
    print("错误: 无法导入 audioldm2 工具。")
    print("请确保已正确安装 'audioldm2-diffusers' 库，或者调整 sys.path 以包含其源代码。")
    # 示例:
    # parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../audioldm2'))
    # sys.path.append(parent_dir)
    sys.exit(1)


# ===================================================================
#               归一化辅助函数
# ===================================================================
import pyloudnorm as pyln

def normalize_peak(waveform, target_peak=0.95):
    """峰值归一化"""
    peak = np.max(np.abs(waveform))
    if peak == 0: return waveform
    return waveform / peak * target_peak

def normalize_rms(waveform, target_rms=0.1):
    """均方根归一化"""
    rms = np.sqrt(np.mean(waveform**2))
    if rms == 0: return waveform
    return waveform / rms * target_rms

def normalize_lufs(waveform, sample_rate, target_lufs=-23.0):
    """响度归一化 (LUFS)"""
    if waveform.dtype != np.float32 and waveform.dtype != np.float64:
        waveform = waveform.astype(np.float32) / np.iinfo(waveform.dtype).max if waveform.dtype in [np.int16, np.int32] else waveform.astype(np.float32)

    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(waveform)
    
    gain_db = target_lufs - loudness
    gain_linear = 10.0 ** (gain_db / 20.0)
    normalized_waveform = waveform * gain_linear
    
    if np.max(np.abs(normalized_waveform)) > 1.0:
        normalized_waveform = normalize_peak(normalized_waveform, target_peak=0.98)
        print("警告: LUFS归一化后可能导致削波，已进行额外的峰值归一化。")

    return normalized_waveform

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
        window = getattr(torch, self.window)(win_length).to(x.device)
        stft = torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window, return_complex=True)
        return torch.abs(stft)
    
    def compute_loss(self, pred, target):
        pred = torch.from_numpy(pred).float() if isinstance(pred, np.ndarray) else pred
        target = torch.from_numpy(target).float() if isinstance(target, np.ndarray) else target
        
        if pred.dim() == 1: pred = pred.unsqueeze(0)
        if target.dim() == 1: target = target.unsqueeze(0)
            
        total_loss = 0.0
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            pred_spec, target_spec = self.stft(pred, fft_size, hop_size, win_length), self.stft(target, fft_size, hop_size, win_length)
            total_loss += F.l1_loss(pred_spec, target_spec) + F.mse_loss(torch.log(pred_spec + 1e-7), torch.log(target_spec + 1e-7))
            
        return total_loss / len(self.fft_sizes)

# --- 参数解析 ---
parser = argparse.ArgumentParser(description='从潜在表示还原音频并计算多维度指标')
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
            waveform_original = waveform_original.squeeze().numpy()[:48000]
            scipy.io.wavfile.write(original_audio_path, rate=16000, data=waveform_original)
            print(f"原始音频已提取并重采样，保存至: {original_audio_path}")
        else:
            print(f"警告：无法从视频提取音频。错误: {result.stderr}")
    finally:
        if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
else:
    print(f"警告：找不到原始视频文件 '{original_video_path}'")

# --- 3. 加载 AudioLDM 2 模型及相关工具 ---
print("\n正在加载 AudioLDM 2 模型...")
repo_id = "cvssp/audioldm2"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)

device = args.device if args.device != 'auto' else ("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipe.to(device)
print(f"模型已加载到 {device} 设备。")

vae = pipe.vae

# (新增) 初始化用于计算 Mel Spectrogram 的 STFT 函数
print("正在初始化 STFT 函数以计算 Mel 域损失...")
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

# --- 4. 加载 Latent 并解码 ---
print(f"\n正在从 '{input_latent_path}' 加载潜在表示并解码...")
latent_np = np.load(input_latent_path)
if latent_np.ndim == 4: latent_np = latent_np.squeeze(0)
latent_tensor = torch.from_numpy(latent_np).to(device, dtype=torch.float16).unsqueeze(0)

with torch.no_grad():
    decoded_mel = vae.decode(latent_tensor).sample
    waveform = pipe.mel_spectrogram_to_waveform(decoded_mel).squeeze().cpu().numpy().astype(np.float32)

waveform = waveform[:48000]
scipy.io.wavfile.write(reconstructed_audio_path, rate=16000, data=waveform)
print(f"重建音频已保存至: '{reconstructed_audio_path}'")

# --- 5. 计算并保存所有指标 ---
if waveform_original is not None:
    print("\n--- 计算综合评估指标 ---")
    
    # 步骤 A: 对原始音频进行 LUFS 归一化以进行公平比较
    waveform_original_normalized = normalize_lufs(waveform_original, sample_rate=16000, target_lufs=-23.0)
    print(f"已应用 [LUFS 归一化], 目标: -23.0 LUFS")
    normalized_original_path = os.path.join(output_dir, f"{base_name}_original_normalized.wav")
    scipy.io.wavfile.write(normalized_original_path, rate=16000, data=waveform_original_normalized)
    print(f"归一化后的原始音频已保存至: {normalized_original_path}")

    # 步骤 B: 对齐长度
    min_len = min(len(waveform), len(waveform_original_normalized))
    waveform_recon_aligned = waveform[:min_len]
    waveform_original_aligned = waveform_original_normalized[:min_len]
    
    # 步骤 C: 计算波形域指标
    print("\n计算波形域指标...")
    l1_loss = np.mean(np.abs(waveform_recon_aligned - waveform_original_aligned))
    l2_loss = np.mean((waveform_recon_aligned - waveform_original_aligned) ** 2)
    signal_power = np.mean(waveform_original_aligned ** 2)
    noise_power = l2_loss
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    correlation = np.corrcoef(waveform_recon_aligned, waveform_original_aligned)[0, 1]
    
    print(f"Waveform L1 Loss: {l1_loss:.6f}")
    print(f"Waveform L2 Loss (MSE): {l2_loss:.6f}")
    print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB")
    print(f"Correlation Coefficient: {correlation:.4f}")

    # 步骤 D: 计算多分辨率频谱图损失
    print("\n计算多分辨率频谱图损失...")
    spec_loss_calculator = MultiResolutionSpectrogramLoss()
    spec_loss = spec_loss_calculator.compute_loss(waveform_recon_aligned, waveform_original_aligned)
    print(f"Multi-Resolution Spectrogram Loss: {spec_loss.item():.6f}")

    # 步骤 E: (新增) 计算梅尔频谱图损失 (最直接的VAE性能指标)
    print("\n计算梅尔频谱图损失...")
    mel_l1_loss, mel_mse_loss = -1.0, -1.0 # 默认值
  
    recon_tensor = torch.from_numpy(waveform_recon_aligned).float().unsqueeze(0).to(device)
    orig_tensor = torch.from_numpy(waveform_original_aligned).float().unsqueeze(0).to(device)
    with torch.no_grad():
        mel_reconstructed = fn_STFT.mel_spectrogram(recon_tensor)
        mel_original = fn_STFT.mel_spectrogram(orig_tensor)
    mel_l1_loss = F.l1_loss(mel_reconstructed, mel_original).item()
    mel_mse_loss = F.mse_loss(mel_reconstructed, mel_original).item()
    print(f"Mel Spectrogram L1 Loss (MAE): {mel_l1_loss:.6f}")
    print(f"Mel Spectrogram L2 Loss (MSE): {mel_mse_loss:.6f}")

    # 步骤 F: 保存所有指标到文件
    metrics_path = os.path.join(output_dir, f"{base_name}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"音频重建质量评估: {base_name}\n" + "="*40 + "\n")
        f.write(f"Waveform Domain Metrics\n" + "-"*25 + "\n")
        f.write(f"L1 Loss (MAE): {l1_loss:.6f}\n")
        f.write(f"L2 Loss (MSE): {l2_loss:.6f}\n")
        f.write(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB\n")
        f.write(f"Correlation Coefficient: {correlation:.4f}\n\n")
        f.write(f"Spectrogram Domain Metrics\n" + "-"*25 + "\n")
        f.write(f"Multi-Resolution Spectrogram Loss: {spec_loss.item():.6f}\n\n")
        f.write(f"Mel Spectrogram Domain Metrics (VAE Direct Output)\n" + "-"*25 + "\n")
        f.write(f"Mel L1 Loss (MAE): {mel_l1_loss:.6f}\n")
        f.write(f"Mel L2 Loss (MSE): {mel_mse_loss:.6f}\n")
    print(f"\n所有指标已保存至: {metrics_path}")

else:
    print("\n警告：无法计算指标，因为原始音频不可用。")

print(f"\n所有文件都保存在: '{output_dir}'")