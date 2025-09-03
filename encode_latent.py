import torch
import torch.multiprocessing as mp
from diffusers import AudioLDM2Pipeline
import torchaudio
import librosa
import os
import numpy as np
from tqdm import tqdm
import traceback
from audioldm2.utils import default_audioldm_config
from audioldm2.utilities.audio.stft import TacotronSTFT
# from audioldm2.utilities.audio.tools import wav_to_fbank
def get_mel_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, magnitudes, phases, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    magnitudes = torch.squeeze(magnitudes, 0).numpy().astype(np.float32)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
    return melspec, magnitudes, energy
def setup_audioldm2_vae(gpu_id, repo_id="cvssp/audioldm2", torch_dtype=torch.float16):
    """
    加载并设置 AudioLDM 2 VAE 模型到指定的GPU上。
    这个函数将在每个独立的工作进程中被调用。
    """
    device = f"cuda:{gpu_id}"
    print(f"[GPU-{gpu_id}]: 正在加载 AudioLDM 2 模型到 {device}...")
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch_dtype, resume_download=True)
    pipe = pipe.to(device)
    print(f"[GPU-{gpu_id}]: 模型已成功加载。")
    return pipe.vae, pipe.feature_extractor, device
def pad_wav(waveform, segment_length):
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length
    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    elif waveform_length < segment_length:
        temp_wav = np.zeros((1, segment_length))
        temp_wav[:, :waveform_length] = waveform
    return temp_wav

def _pad_spec(fbank, target_length=1024):
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    if fbank.size(-1) % 2 != 0:
        fbank = fbank[..., :-1]

    return fbank

def normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5
def read_wav_file(filename, segment_length):
    # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
    waveform, sr = torchaudio.load(filename,format="mp4",backend="ffmpeg")  # Faster!!!
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    waveform = waveform.numpy()[0, ...]
    waveform = normalize_wav(waveform)
    waveform = waveform[None, ...]
    waveform = pad_wav(waveform, segment_length)

    # waveform = waveform / np.max(np.abs(waveform))
    # waveform = 0.5 * waveform

    return waveform
def wav_to_fbank(filename, target_length=1024, fn_STFT=None):
    assert fn_STFT is not None

    # mixup
    waveform = read_wav_file(filename, target_length * 160)  # hop size is 160

    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)

    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)

    fbank = torch.FloatTensor(fbank.T)
    log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)

    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )

    return fbank, log_magnitudes_stft, waveform


def encode_audio_from_video(video_path, vae, feature_extractor, device):
    """
    从单个视频文件中提取音频，并使用给定的 VAE 将其编码为潜在表示。
    """
    # 创建一个基于进程和文件名的唯一临时文件，避免冲突
    # temp_audio_path = f"temp_audio_{os.getpid()}_{os.path.basename(video_path)}.wav"
    try:
 
        config=default_audioldm_config()
        # 2. 加载并预处理提取的音频
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
            video_path, target_length=int(duration * 102.4), fn_STFT=fn_STFT
        )
        mel=mel.unsqueeze(0).unsqueeze(0).to(device).to(torch.float16)

        # 3. 使用 VAE 进行编码
        with torch.no_grad():
            latent_representation  = vae.encode(mel).latent_dist.mode()
        
        return latent_representation.cpu().numpy(), "SUCCESS"

    except Exception as e:
        error_message = f"处理文件 '{os.path.basename(video_path)}' 时发生错误: {e}\n{traceback.format_exc()}"
        return None, error_message
    # finally:
    #     # 清理临时文件
    #     if os.path.exists(temp_audio_path):
    #         try:
    #             os.remove(temp_audio_path)
    #         except OSError:
    #             pass

def process_files_on_gpu(gpu_id, file_chunk, input_dir, output_dir):
    """
    这是一个工作函数，由单个进程执行，负责处理分配给它的一批文件。
    """
    # --- 1. 在当前进程中设置模型 ---
    try:
        vae, feature_extractor, device = setup_audioldm2_vae(gpu_id)
    except Exception as e:
        print(f"[GPU-{gpu_id}]: 模型加载失败: {e}")
        return len(file_chunk), 0 # 返回错误，表示所有文件都失败了

    error_count = 0
    success_count = 0

    # --- 2. 遍历并处理分配给这个进程的文件 ---
    # 为每个GPU创建一个独立的进度条
    progress_bar = tqdm(file_chunk, desc=f"GPU-{gpu_id} 处理中", position=gpu_id)
    
    for filename in progress_bar:
        video_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}.npy")

        # if os.path.exists(output_path):
        #     continue

        latent_np, status = encode_audio_from_video(video_path, vae, feature_extractor, device)

        if status == "SUCCESS" and latent_np is not None:
            np.save(output_path, latent_np)
            success_count += 1
        elif status != "NO_AUDIO":
            # 如果不是因为没有音轨而失败，就打印错误并计数
            tqdm.write(f"[GPU-{gpu_id} 错误]: {status}")
            error_count += 1
            
    return error_count, success_count

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
        
    # video_directory_list=["vggsound_00_3s","vggsound_01_3s","vggsound_02_3s","vggsound_03_3s","vggsound_04_3s"]
    video_directory_list=["vggsound_05_3s","vggsound_06_3s","vggsound_07_3s","vggsound_08_3s","vggsound_09_3s","vggsound_10_3s","vggsound_11_3s","vggsound_12_3s","vggsound_13_3s","vggsound_14_3s"]
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