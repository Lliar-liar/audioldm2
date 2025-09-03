import torch
from diffusers import AudioLDM2Pipeline
import numpy as np
from scipy.io.wavfile import write
import os
import scipy
import argparse

# --- 参数解析 ---
parser = argparse.ArgumentParser(description='从潜在表示还原音频')
parser.add_argument('--input_latent_path', type=str, required=True,
                    help='输入的 latent npy 文件路径')
parser.add_argument('--output_audio_path', type=str, default='/blob/avtok/recon.wav',
                    help='输出的还原音频文件路径 (默认: /blob/avtok/recon.wav)')
parser.add_argument('--device', type=str, default='auto',
                    choices=['cuda', 'cpu', 'auto'],
                    help='运行设备 (默认: auto)')
args = parser.parse_args()

# --- 1. 设置文件路径 ---
input_latent_path = args.input_latent_path
output_audio_path = args.output_audio_path

# --- 2. 检查输入文件是否存在 ---
if not os.path.exists(input_latent_path):
    raise FileNotFoundError(f"错误：找不到 latent 文件 '{input_latent_path}'。请先运行编码脚本。")

# --- 3. 加载 AudioLDM 2 模型 ---
print("正在加载 AudioLDM 2 模型...")
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

# --- 4. 加载 Latent 并解码 ---
print(f"\n正在从 '{input_latent_path}' 加载潜在表示...")
latent_np = np.load(input_latent_path)
latent_tensor = torch.from_numpy(latent_np).to(device, dtype=torch.float16)

print("开始解码过程...")

# 在不计算梯度的模式下进行推理
with torch.no_grad():
    # --- 步骤 4a: 使用 VAE 解码器 ---
    print("步骤 1/2: 使用 VAE 解码器将潜在表示转为梅尔频谱图...")
    decoded_mel = vae.decode(latent_tensor).sample
    
    print("步骤 2/2: 将梅尔频谱图转换为波形...")
    waveform = pipe.mel_spectrogram_to_waveform(decoded_mel)
    waveform = waveform.squeeze().detach().cpu().numpy().astype(np.float32)

# 保存音频
scipy.io.wavfile.write(output_audio_path, rate=16000, data=waveform)

print("\n--- 操作成功 ---")
print(f"从 latent 文件还原的音频已成功保存为 '{output_audio_path}'。")
