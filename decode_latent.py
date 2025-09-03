import torch
from diffusers import AudioLDM2Pipeline

import numpy as np
from scipy.io.wavfile import write
import os


# --- 1. 设置文件路径 ---
# 要读取的 latent npy 文件
input_latent_path = "/blob/vggsound_cropped_audio_latent/vggsound_00_3s/0s49D-LqHwg_000030_part_003.npy"
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




# print(f"原始 Latent Tensor 形状: {latent_tensor.shape}")

# # 1. 计算截断长度
# #    我们操作的是第3个维度 (索引为2)，也就是时间维度
# original_time_len = latent_tensor.shape[2]
# truncation_len = int(original_time_len * 3 / 10) # 保留的长度

# # 添加一个安全检查，如果截断长度为0，则不进行任何操作
# if truncation_len > 0:
#     print(f"将在时间维度 (dim=2) 上保留前 {truncation_len} 个单位，并用这部分进行循环填充。")

#     # 2. 截取 tensor 的前 3/10 部分，这将是我们的重复“模式”
#     pattern = latent_tensor[:, :, :truncation_len, :]
#     print(f"用于循环填充的 'pattern' 形状: {pattern.shape}")

#     # 3. 计算需要重复多少次才能填满或超过原始长度
#     #    这相当于向上取整除法: ceil(original_time_len / truncation_len)
#     num_repeats = (original_time_len + truncation_len - 1) // truncation_len
#     print(f"'pattern' 将被重复 {num_repeats} 次。")
    
#     # 4. 沿着时间维度 (dim=2) 重复这个 pattern
#     #    torch.repeat 的参数是每个维度重复的次数
#     #    我们只想在时间维度上重复，所以其他维度都是1
#     repeated_latent = pattern.repeat(1, 1, num_repeats, 1)
#     print(f"重复后的中间张量形状: {repeated_latent.shape}")

#     # 5. 最后，从这个重复后的长张量中，截取出与原始长度完全相同的部分
#     #    这样可以确保即使重复后超长了，最终尺寸也是正确的
#     final_latent_tensor = repeated_latent[:, :, :original_time_len, :]

#     # 用这个新张量覆盖原始的 latent_tensor 变量
#     latent_tensor = final_latent_tensor
    
#     print(f"循环填充后的 Latent Tensor 形状: {latent_tensor.shape}")

# else:
#     print("警告：计算出的截断长度为0，无法进行循环填充。将使用原始 Latent Tensor。")




print("开始解码过程...")

# 在不计算梯度的模式下进行推理
with torch.no_grad():
    # --- 步骤 4a: 使用 VAE 解码器 ---
    print("步骤 1/2: 使用 VAE 解码器将潜在表示转为梅尔频谱图...")
    decoded_mel = vae.decode(latent_tensor).sample
    print(f"VAE 解码后得到的原始梅尔频谱图形状: {decoded_mel.shape}")
    
    # --- 步骤 4b: 重塑梅尔频谱图以适配声码器 (修正部分) ---
    #
    # *** 关键修正点 ***
    # 声码器期望输入为 [Batch, Channels, Length] (3D)
    # 而 VAE 输出可能是 [Batch, 1, 1, Length, Channels] (5D)
    # 我们需要移除多余的维度并调整顺序
    #
    # 1. 移除所有大小为 1 的维度
    mel_squeezed = decoded_mel.squeeze()
    
    # 2. 如果 squeeze 后是 2D [Length, Channels]，增加 Batch 维度
    if mel_squeezed.dim() == 2:
        mel_squeezed = mel_squeezed.unsqueeze(0)
        
    
    print(f"重塑后送入声码器的梅尔频谱图形状: {mel_reshaped.shape}")

    # --- 步骤 4c: 使用声码器生成波形 ---
    print("步骤 2/2: 使用声码器将梅尔频谱图合成为音频波形...")
    # 确保调用时没有 return_dict 参数
    waveform = pipe.mel_spectrogram_to_waveform(mel_squeezed)
    waveform =waveform.squeeze().detach().cpu().numpy().astype(np.float32)

scipy.io.wavfile.write(output_audio_path, rate=SAMPLING_RATE, data=waveform)



print("\n--- 操作成功 ---")
print(f"从 latent 文件还原的音频已成功保存为 '{output_audio_path}'。")