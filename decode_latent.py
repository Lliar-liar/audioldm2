import torch
from diffusers import AudioLDM2Pipeline

import numpy as np
from scipy.io.wavfile import write
import os
import scipy


# --- 1. 设置文件路径 ---
# 要读取的 latent npy 文件
input_latent_path = "/blob/vggsound_cropped_audio_latent/vggsound_05_3s/FBkx6WalE7Y_000300_part_002.npy"
# 输出的还原音频文件
# output_audio_path = "reconstructed_from_latent.wav"
output_audio_path = "/blob/avtok/recon.wav"

# --- 2. 检查输入文件是否存在 ---
if not os.path.exists(input_latent_path):
    raise FileNotFoundError(f"错误：找不到 latent 文件 '{input_latent_path}'。请先运行编码脚本。")

# --- 3. 加载 AudioLDM 2 模型 ---
print("正在加载 AudioLDM 2 模型...")
repo_id = "cvssp/audioldm2"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
print(pipe.feature_extractor.sampling_rate)

# 如果有可用的 GPU，则将模型移至 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
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

    waveform = pipe.mel_spectrogram_to_waveform(decoded_mel)
    waveform =waveform.squeeze().detach().cpu().numpy().astype(np.float32)

scipy.io.wavfile.write(output_audio_path, rate=16000, data=waveform)



print("\n--- 操作成功 ---")
print(f"从 latent 文件还原的音频已成功保存为 '{output_audio_path}'。")