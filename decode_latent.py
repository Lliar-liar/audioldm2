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
            
            # 保存原始音频（重采样后）
            scipy.io.wavfile.write(original_audio_path, rate=16000, data=waveform_original)
            print(f"原始音频已提取并重采样，保存至: {original_audio_path}")
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
latent_tensor = torch.from_numpy(latent_np).to(device, dtype=torch.float16)

print("开始解码过程...")

# 在不计算梯度的模式下进行推理
with torch.no_grad():
    # --- 步骤 6a: 使用 VAE 解码器 ---
    print("步骤 1/2: 使用 VAE 解码器将潜在表示转为梅尔频谱图...")
    decoded_mel = vae.decode(latent_tensor).sample
    
    print("步骤 2/2: 将梅尔频谱图转换为波形...")
    waveform = pipe.mel_spectrogram_to_waveform(decoded_mel)
    waveform = waveform.squeeze().detach().cpu().numpy().astype(np.float32)

# 保存重建音频
scipy.io.wavfile.write(reconstructed_audio_path, rate=16000, data=waveform)

print("\n--- 操作成功 ---")
print(f"重建音频已保存至: '{reconstructed_audio_path}'")
if os.path.exists(original_audio_path):
    print(f"原始音频已保存至: '{original_audio_path}'")
print(f"所有文件都保存在: '{output_dir}'")
