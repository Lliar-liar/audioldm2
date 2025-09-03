import torch
import torch.multiprocessing as mp
from diffusers import AudioLDM2Pipeline
import moviepy.editor as mp_editor
import librosa
import os
import numpy as np
from tqdm import tqdm
import traceback
from audioldm2.utils import default_audioldm_config
from audioldm2.utilities.audio.stft import TacotronSTFT

# --- 优化点: 将配置和STFT对象创建移至每个进程的顶层 ---
# 将配置设为全局，方便所有进程访问
config = default_audioldm_config()

def setup_audioldm2_vae(gpu_id, repo_id="cvssp/audioldm2", torch_dtype=torch.float16):
    """
    加载并设置 AudioLDM 2 VAE 模型到指定的GPU上。
    """
    device = f"cuda:{gpu_id}"
    print(f"[GPU-{gpu_id}]: 正在加载 AudioLDM 2 模型到 {device}...")
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch_dtype, resume_download=True)
    pipe = pipe.to(device)
    print(f"[GPU-{gpu_id}]: 模型已成功加载。")
    # feature_extractor 在新版本中通常不是必需的
    return pipe.vae, device

# ##################################################################
# ## 以下是经过修正和优化的音频处理辅助函数 ##
# ##################################################################

def _pad_spec(fbank: torch.Tensor, target_length: int):
    """
    【修正版】辅助函数：将频谱图补齐或裁剪到目标长度。全程使用PyTorch操作。
    """
    n_frames = fbank.shape[0]
    if n_frames > target_length:
        # 长度超出，则裁剪
        fbank = fbank[:target_length, :]
    elif n_frames < target_length:
        # 长度不足，则用0补齐
        pad_width = target_length - n_frames
        # 使用 torch.nn.functional.pad，参数格式为 (左, 右, 上, 下)
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, pad_width), mode='constant', value=0)
    return fbank

def get_mel_from_wav(waveform: torch.Tensor, _stft: TacotronSTFT):
    """
    【修正版】辅助函数：从波形Tensor计算梅尔频谱。全程保持在PyTorch中，不转回NumPy。
    """
    # 确保输入是正确的形状 [batch_size, n_samples]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # 直接调用STFT对象的mel_spectrogram方法，其输入和输出都是Tensor
    melspec, magnitudes, _, _ = _stft.mel_spectrogram(waveform)
    
    # 移除批次维度，返回纯Tensor
    return melspec.squeeze(0), magnitudes.squeeze(0)

def waveform_to_fbank(
    waveform: torch.Tensor,
    target_length: int,
    fn_STFT: TacotronSTFT
):
    """
    【修正版】优化函数，将波形Tensor高效转换为fbank。
    """
    assert torch.is_tensor(waveform), "输入 'waveform' 必须是 PyTorch tensor."
    
    # 1. 计算梅尔频谱，全程使用Tensor
    fbank_T, _ = get_mel_from_wav(waveform, fn_STFT)

    # 2. 转置维度 [n_mels, time] -> [time, n_mels]
    fbank = fbank_T.T

    # 3. 补齐或裁剪
    fbank = _pad_spec(fbank, target_length)

    return fbank, None # 保持与原函数相同的返回格式

# ##################################################################
# ## 核心修改：重写 encode_audio_from_video 函数 ##
# ##################################################################

def encode_audio_from_video(
    video_path: str,
    vae,
    device: str,
    fn_STFT: TacotronSTFT # 接收预先创建好的STFT处理器
):
    """
    【优化版】从视频文件中提取音频，并将其编码为潜在表示。
    此版本在内存中完成所有操作，避免了磁盘I/O，并增加了数据校验。
    """
    try:
        with mp_editor.VideoFileClip(video_path) as video_clip:
            if video_clip.audio is None:
                return None, "NO_AUDIO"
            
            # 1. 【优化】直接将音频解码到内存中的NumPy数组
            audio_array = video_clip.audio.to_soundarray(fps=config["preprocessing"]["audio"]["sampling_rate"])
            
            # 如果是立体声，则混合为单声道
            if audio_array.ndim > 1 and audio_array.shape[1] > 1:
                audio_array = np.mean(audio_array, axis=1)

        # 2. 【鲁棒性】在处理前检查是否存在损坏的音频数据 (NaN 或 Inf)
        if np.isnan(audio_array).any() or np.isinf(audio_array).any():
            return None, f"INVALID_AUDIO (NaN/Inf) in {os.path.basename(video_path)}"

        # 3. 【优化】将NumPy数组转换为PyTorch Tensor（这是唯一必要的转换）
        waveform = torch.from_numpy(audio_array.copy()).float()
        
        # 4. 【优化】调用高效的内存计算函数
        mel, _ = waveform_to_fbank(
            waveform=waveform,
            target_length=int(3 * 102.4),  # 假设时长为3秒
            fn_STFT=fn_STFT
        )
        
        # 5. 准备输入给VAE的Tensor（增加批次和通道维度，并移动到GPU）
        mel = mel.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float16)

        # 6. 使用VAE进行编码
        with torch.no_grad():
            latent_representation = vae.encode(mel).latent_dist.mode()
        
        return latent_representation.cpu().numpy(), "SUCCESS"

    except Exception as e:
        error_message = f"处理文件 '{os.path.basename(video_path)}' 时出错: {e}\n{traceback.format_exc()}"
        return None, error_message
    # 不再需要 finally 块，因为没有临时文件了

def process_files_on_gpu(gpu_id, file_chunk, input_dir, output_dir):
    """
    工作函数，由单个进程执行。
    """
    try:
        vae, device = setup_audioldm2_vae(gpu_id)
    except Exception as e:
        print(f"[GPU-{gpu_id}]: 模型加载失败: {e}")
        return len(file_chunk), 0

    # --- 优化点: 在每个进程中只创建一次STFT处理器 ---
    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    error_count = 0
    success_count = 0
    progress_bar = tqdm(file_chunk, desc=f"GPU-{gpu_id} 处理中", position=gpu_id)
    
    for filename in progress_bar:
        video_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}.npy")

        # 将预先创建好的 fn_STFT 对象传递下去
        latent_np, status = encode_audio_from_video(video_path, vae, device, fn_STFT)

        if status == "SUCCESS" and latent_np is not None:
            np.save(output_path, latent_np)
            success_count += 1
        elif status not in ["SUCCESS", "NO_AUDIO"]:
            tqdm.write(f"[GPU-{gpu_id} 已跳过]: 文件: {filename}, 原因: {status}")
            error_count += 1
            
    return error_count, success_count

# `batch_process_videos_multi_gpu` 和 `if __name__ == '__main__':` 部分保持不变
# ... (您的这部分代码是正确的，无需修改)
def batch_process_videos_multi_gpu(input_dir, output_dir):
    """
    主函数：负责任务分发和启动多进程。
    """
    # --- 1. 检查GPU数量 ---
    if not torch.cuda.is_available():
        print("错误：未检测到CUDA设备。请在单卡CPU/GPU模式下运行。")
        return

    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 块可用的GPU。")

    # --- 2. 检查并创建输出目录 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # --- 3. 查找所有要处理的 MP4 文件 ---
    all_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")])
    if not all_files:
        print(f"在目录 '{input_dir}' 中没有找到 .mp4 文件。")
        return
        
    print(f"在输入目录中找到 {len(all_files)} 个 .mp4 文件。准备分发任务...")

    # --- 4. 将文件列表平均分配给每个GPU ---
    # 使用 np.array_split 可以优雅地处理无法整除的情况
    file_chunks = np.array_split(all_files, num_gpus)
    
    # 准备传递给每个工作进程的参数
    tasks = []
    for gpu_id, chunk in enumerate(file_chunks):
        if len(chunk) > 0: # 只有当分配到文件时才创建任务
            tasks.append((gpu_id, list(chunk), input_dir, output_dir))
            print(f"  -> GPU-{gpu_id} 将处理 {len(chunk)} 个文件。")

    # --- 5. 创建并启动进程池 ---
    # 使用 'spawn' 启动方法，这对CUDA是必须的，可以避免很多潜在的死锁问题
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=len(tasks)) as pool:
        # 使用 starmap 来传递多个参数给工作函数
        results = pool.starmap(process_files_on_gpu, tasks)

    # --- 6. 汇总结果 ---
    total_errors = sum([res[0] for res in results])
    total_success = sum([res[1] for res in results])

    print("\n--- 处理完成，生成报告 ---")
    print(f"总计成功处理: {total_success} 个文件")
    print(f"总计处理失败: {total_errors} 个文件")
    print("\n🎉 所有视频处理完成！")


if __name__ == '__main__':
    # 为了在使用CUDA时获得最佳的多进程稳定性，建议设置启动方法
    # 'spawn' 会创建一个全新的Python解释器进程，而不是'fork'一个现有进程
    # 这可以避免CUDA初始化状态在子进程中出现问题。
    # 必须在 if __name__ == '__main__': 块的开头设置。
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    video_directory_list=["vggsound_00_3s","vggsound_01_3s","vggsound_02_3s","vggsound_03_3s","vggsound_04_3s"]
    input_video_directory_base="/blob/vggsound_cropped"
    output_latent_directory_base = "/blob/vggsound_cropped_audio_latent"
    # --------------------------
    for video_dir in video_directory_list:
        input_video_directory=os.path.join(input_video_directory_base,video_dir)
        output_latent_directory=os.path.join(output_latent_directory_base,video_dir)

        try:
            batch_process_videos_multi_gpu(input_video_directory, output_latent_directory)
        except Exception as e:
            print(f"\n程序运行期间发生严重错误: {e}")
            print(traceback.format_exc())