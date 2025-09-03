import torch
import librosa
import numpy as np
import scipy.io.wavfile
from diffusers import AudioLDM2Pipeline

from audioldm2.utils import default_audioldm_config
from audioldm2.utilities.audio.stft import TacotronSTFT
from audioldm2.utilities.audio.tools import wav_to_fbank
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../audioldm2'))

sys.path.append(parent_dir)


def reconstruct_audio_with_vae(
    model_id: str,
    input_wav: str,
    output_wav: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float32
):
    """
    使用 AudioLDM2 的 VAE 对音频文件进行编码和解码。

    Args:
        model_id (str): 要使用的 Hugging Face 模型ID (例如 "cvssp/audioldm2")。
        input_wav (str): 输入的WAV文件路径。
        output_wav (str): 保存重建后音频的WAV文件路径。
        device (str): 运行模型的设备 ("cuda" 或 "cpu")。
        dtype (torch.dtype): 运行模型的精度 (torch.float16 或 torch.float32)。
    """
    print(f"正在使用设备: {device}")

    # 1. 加载预训练的 AudioLDM2 Pipeline
    # ------------------------------------------------
    # 这个pipeline包含了我们需要的所有组件：VAE, 声码器 (vocoder) 等。
    print("正在加载 AudioLDM2 pipeline...")
    try:
        pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保已安装所有依赖项: pip install torch diffusers transformers accelerate librosa scipy")
        return

    vae = pipe.vae
    print(vae.quant_conv)
    vocoder = pipe.vocoder
    config=default_audioldm_config()
    print(config)
    # 获取模型的采样率和梅尔频谱的维度
    SAMPLING_RATE = vocoder.config.sampling_rate
    N_MELS = vocoder.config.model_in_dim

    print(f"模型期望的采样率为: {SAMPLING_RATE} Hz")
    print(f"模型期望的梅尔频谱维度为: {N_MELS}")


    # 2. 加载并预处理音频
    # ------------------------------------------------
    # a. 使用librosa加载音频，并重采样到模型所期望的采样率
    print(f"正在加载音频文件: {input_wav}")
    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )
    duration=3
    # waveform = read_wav_file(original_audio_file_path, None)
    mel, _, _ = wav_to_fbank(
        input_wav, target_length=int(duration * 102.4), fn_STFT=fn_STFT
    )
    print(f"声谱图Tensor的形状: {mel.shape}")

    mel=mel.unsqueeze(0).unsqueeze(0).to(device)
    print("正在使用 VAE 编码...")
    with torch.no_grad():
        # .latent_dist 包含了潜在空间的分布（均值和方差）
        latents = vae.encode(mel).latent_dist.mode()
    print(latents.shape)

    # original_length = latents.shape[2]
    # new_length = int(original_length * 3 / 10)
    # truncated_latents = latents[:, :, :new_length, :]
    # num_repeats = original_length // new_length
    # remainder = original_length % new_length

    # # Repeat the truncated tensor fully as many times as possible
    # repeated_tensors = [truncated_latents] * num_repeats

    # # Add the remaining part from the beginning of the truncated tensor
    # if remainder > 0:
    #     remainder_tensor = truncated_latents[:, :, :remainder, :]
    #     repeated_tensors.append(remainder_tensor)

    # # Concatenate the parts along the third dimension
    # reconstructed_latents = torch.cat(repeated_tensors, dim=2)
    # latents=reconstructed_latents

    print("正在使用 VAE 解码...")
    with torch.no_grad():
        reconstructed_spec_tensor = vae.decode(latents).sample

    print(reconstructed_spec_tensor.shape)

    print("正在使用声码器将声谱图转换回音频...")
    # 该辅助函数会处理所有必要的步骤
    reconstructed_spec_tensor=reconstructed_spec_tensor
    reconstructed_waveform = pipe.mel_spectrogram_to_waveform(reconstructed_spec_tensor)
    
    # 裁剪音频长度以匹配原始输入
    # reconstructed_waveform = reconstructed_waveform[:, :waveform.shape[0]]


    # 6. 保存重建的音频
    # ------------------------------------------------
    # 将Torch Tensor转换为Numpy数组以便保存。
    reconstructed_waveform_np = reconstructed_waveform.squeeze().detach().cpu().numpy().astype(np.float32)
    
    # 保存为WAV文件
    print(f"正在保存重建后的音频到: {output_wav}")
    scipy.io.wavfile.write(output_wav, rate=SAMPLING_RATE, data=reconstructed_waveform_np)
    print("完成！")


if __name__ == '__main__':
    # --- 配置参数 ---
    # 请将 "path/to/your/audio.wav" 替换成你的WAV文件路径
    INPUT_AUDIO_FILE = "/home/yifanyang/ziweizhou/output/02_09_2025_02_46_32/Musical constellations twinkling in the night sky, forming a cosmic melody..wav" 
    
    # 重建后文件的保存路径
    OUTPUT_AUDIO_FILE = "/home/yifanyang/ziweizhou/output/02_09_2025_02_46_32/recon.wav"
    
    # 使用的模型ID，"cvssp/audioldm2" 是一个通用的文生音模型
    MODEL_REPO_ID = "cvssp/audioldm2"

    # 检查输入文件是否存在
    import os
    if not os.path.exists(INPUT_AUDIO_FILE):
        print(f"错误: 输入文件 '{INPUT_AUDIO_FILE}' 不存在。")
        print("请下载一个示例WAV文件或修改 INPUT_AUDIO_FILE 变量的路径。")
    else:
        reconstruct_audio_with_vae(
            model_id=MODEL_REPO_ID,
            input_wav=INPUT_AUDIO_FILE,
            output_wav=OUTPUT_AUDIO_FILE
        )