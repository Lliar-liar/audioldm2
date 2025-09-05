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

# MultiResolutionSpectrogramLoss implementation
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
        magnitude = torch.abs(stft)
        return magnitude
    
    def compute_loss(self, pred, target):
        """Compute multi-resolution spectrogram loss"""
        pred = torch.from_numpy(pred).float() if isinstance(pred, np.ndarray) else pred
        target = torch.from_numpy(target).float() if isinstance(target, np.ndarray) else target
        
        # Ensure same shape
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)
            
        total_loss = 0.0
        
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            pred_spec = self.stft(pred, fft_size, hop_size, win_length)
            target_spec = self.stft(target, fft_size, hop_size, win_length)
            
            # L1 loss on magnitude
            l1_loss = F.l1_loss(pred_spec, target_spec)
            
            # L2 loss on log magnitude
            pred_log = torch.log(pred_spec + 1e-7)
            target_log = torch.log(target_spec + 1e-7)
            l2_loss = F.mse_loss(pred_log, target_log)
            
            total_loss += l1_loss + l2_loss
            
        return total_loss / len(self.fft_sizes)

# --- 参数解析 ---
parser = argparse.ArgumentParser(description='从潜在表示还原音频')
parser.add_argument('--input_latent_path', type=str, required=True,
                    help='输入的 latent npy 文件路径')
parser.add_argument('--output_dir', type=str, default='/blob/avtok/',
                    help='输出目录 (默认: /blob/avtok/)')
parser.add_argument('--device', type=str, default='auto',
                    choices=['cuda', 'cpu', 'auto'],
                    help='运行设备 (默认: auto)')
args = parser.parse_args()

origin_video_base_dir = "/blob/vggsound_cropped/"
latent_base_dir = "/blob/vggsound_cropped_audio_latent_fixed/"

# --- 1. 设置文件路径 ---
input_latent_path = args.input_latent_path
output_dir = args.output_dir

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# --- 2. 检查输入文件是否存在 ---
if not os.path.exists(input_latent_path):
    raise FileNotFoundError(f"错误：找不到 latent 文件 '{input_latent_path}'。请先运行编码脚本。")

# --- 3. 根据latent路径构建原始视频路径 ---
# 从 latent 路径中提取相对路径部分
relative_path = input_latent_path.replace(latent_base_dir, "")
# 将 .npy 替换为 .mp4
video_relative_path = relative_path.replace(".npy", ".mp4")
# 构建完整的视频路径
original_video_path = os.path.join(origin_video_base_dir, video_relative_path)

# 获取文件名（不含扩展名）作为基础名
base_name = os.path.splitext(os.path.basename(input_latent_path))[0]

# 设置输出文件名
reconstructed_audio_path = os.path.join(output_dir, f"{base_name}_reconstructed.wav")
original_audio_path = os.path.join(output_dir, f"{base_name}_original.wav")

print(f"原始视频路径: {original_video_path}")
print(f"重建音频将保存至: {reconstructed_audio_path}")
print(f"原始音频将保存至: {original_audio_path}")

# --- 4. 从原始视频提取音频并重采样 ---
waveform_original = None
if os.path.exists(original_video_path):
    print(f"\n正在从视频提取并重采样原始音频...")
    
    # 创建临时文件用于存储提取的音频
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
    
    try:
        # 使用 ffmpeg 提取音频并重采样到 16000Hz
        cmd = [
            'ffmpeg', '-i', original_video_path,
            '-vn',  # 不要视频
            '-ar', '16000',  # 重采样到 16000Hz
            '-ac', '1',  # 单声道
            '-f', 'wav',
            '-y',  # 覆盖输出文件
            temp_audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # 读取重采样后的音频
            waveform_original, sr = torchaudio.load(temp_audio_path)
            waveform_original = waveform_original.squeeze().numpy()
            
            # 截取前3秒（48000个样本点）
            waveform_original = waveform_original[:48000]
            
            # 保存原始音频（重采样后）
            scipy.io.wavfile.write(original_audio_path, rate=16000, data=waveform_original)
            print(f"原始音频已提取并重采样，保存至: {original_audio_path}")
            print(f"原始音频形状: {waveform_original.shape}")
        else:
            print(f"警告：无法从视频提取音频。错误信息: {result.stderr}")
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
else:
    print(f"警告：找不到原始视频文件 '{original_video_path}'")

# --- 5. 加载 AudioLDM 2 模型 ---
print("\n正在加载 AudioLDM 2 模型...")
repo_id = "cvssp/audioldm2"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
print(f"采样率: {pipe.feature_extractor.sampling_rate}")

# 设置设备
if args.device == 'auto':
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = args.device

pipe = pipe.to(device)
print(f"模型已加载到 {device} 设备。")

# 获取 VAE 和声码器
vae = pipe.vae
vocoder = pipe.vocoder

# --- 6. 加载 Latent 并解码 ---
print(f"\n正在从 '{input_latent_path}' 加载潜在表示...")
latent_np = np.load(input_latent_path)
latent_tensor = torch.from_numpy(latent_np).to(device, dtype=torch.float16).unsqueeze(0)
print(latent_tensor.shape)
print("开始解码过程...")

# 在不计算梯度的模式下进行推理
with torch.no_grad():
    # --- 步骤 6a: 使用 VAE 解码器 ---
    print("步骤 1/2: 使用 VAE 解码器将潜在表示转为梅尔频谱图...")
    decoded_mel = vae.decode(latent_tensor).sample
    
    print("步骤 2/2: 将梅尔频谱图转换为波形...")
    waveform = pipe.mel_spectrogram_to_waveform(decoded_mel)
    waveform = waveform.squeeze().detach().cpu().numpy().astype(np.float32)

print(f"重建音频形状: {waveform.shape}")
waveform = waveform[:48000]  # 截取前3秒

# 保存重建音频
scipy.io.wavfile.write(reconstructed_audio_path, rate=16000, data=waveform)

print("\n--- 操作成功 ---")
print(f"重建音频已保存至: '{reconstructed_audio_path}'")
if os.path.exists(original_audio_path):
    print(f"原始音频已保存至: '{original_audio_path}'")

# --- 7. 计算损失指标 ---
if waveform_original is not None:
    print("\n--- 计算损失指标 ---")
    
    # 确保两个音频长度相同
    min_length = min(len(waveform), len(waveform_original))
    waveform_recon_aligned = waveform[:min_length]
    waveform_original_aligned = waveform_original[:min_length]
    
    # 1. 计算 L1 Loss (Waveform)
    l1_loss = np.mean(np.abs(waveform_recon_aligned - waveform_original_aligned))
    print(f"Waveform L1 Loss: {l1_loss:.6f}")
    
    # 2. 计算 L2 Loss (Waveform) - 额外信息
    l2_loss = np.mean((waveform_recon_aligned - waveform_original_aligned) ** 2)
    print(f"Waveform L2 Loss (MSE): {l2_loss:.6f}")
    
    # 3. 计算 Multi-Resolution Spectrogram Loss
    print("\n计算 Multi-Resolution Spectrogram Loss...")
    spec_loss_calculator = MultiResolutionSpectrogramLoss()
    
    # 将numpy数组转换为tensor
    waveform_recon_tensor = torch.from_numpy(waveform_recon_aligned).float()
    waveform_original_tensor = torch.from_numpy(waveform_original_aligned).float()
    
    spec_loss = spec_loss_calculator.compute_loss(waveform_recon_tensor, waveform_original_tensor)
    print(f"Multi-Resolution Spectrogram Loss: {spec_loss.item():.6f}")
    
    # 4. 计算信噪比 (SNR)
    signal_power = np.mean(waveform_original_aligned ** 2)
    noise_power = np.mean((waveform_recon_aligned - waveform_original_aligned) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB")
    
    # 5. 计算相关系数
    correlation = np.corrcoef(waveform_recon_aligned, waveform_original_aligned)[0, 1]
    print(f"Correlation Coefficient: {correlation:.4f}")
    
    # 保存损失指标到文件
    metrics_path = os.path.join(output_dir, f"{base_name}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"音频重建质量评估指标\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"文件: {base_name}\n")
        f.write(f"原始音频长度: {len(waveform_original_aligned)} samples\n")
        f.write(f"重建音频长度: {len(waveform_recon_aligned)} samples\n")
        f.write(f"\n损失指标:\n")
        f.write(f"Waveform L1 Loss: {l1_loss:.6f}\n")
        f.write(f"Waveform L2 Loss (MSE): {l2_loss:.6f}\n")
        f.write(f"Multi-Resolution Spectrogram Loss: {spec_loss.item():.6f}\n")
        f.write(f"\n质量指标:\n")
        f.write(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB\n")
        f.write(f"Correlation Coefficient: {correlation:.4f}\n")
    
    print(f"\n指标已保存至: {metrics_path}")
else:
    print("\n警告：无法计算损失指标，因为原始音频不可用。")

print(f"\n所有文件都保存在: '{output_dir}'")
