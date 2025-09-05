import os
import sys
import argparse
import tempfile
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import scipy
from diffusers import AudioLDM2Pipeline

# 需要安装 pyloudnorm 用于 LUFS 度量与归一化
# pip install pyloudnorm
try:
    import pyloudnorm as pyln
except ImportError as e:
    raise ImportError("本脚本需要 'pyloudnorm' 来进行 LUFS 正则化与指标计算。请先安装: pip install pyloudnorm") from e


def dbfs(x: np.ndarray, eps: float = 1e-12) -> float:
    """返回样本峰值对应的 dBFS（0 dBFS 对应幅度 1.0）。"""
    peak = float(np.max(np.abs(x))) if x.size > 0 else 0.0
    if peak <= eps:
        return -np.inf
    return 20.0 * np.log10(peak + 1e-12)


def compute_loudness_metrics(x: np.ndarray, sr: int):
    """计算一组基础响度指标：Integrated LUFS, Loudness Range (LRA), sample peak dBFS, clipping 计数。"""
    x = x.astype(np.float32)
    meter = pyln.Meter(sr)  # EBU R128
    lufs = float(meter.integrated_loudness(x))
    lra = float(meter.loudness_range(x))
    peak_db = dbfs(x)
    # 简单的采样点削波检测（非真峰值），可按需放宽阈值
    clip_count = int(np.sum(np.abs(x) >= 0.9995))
    return {
        "lufs": lufs,
        "lra": lra,
        "peak_dbfs": peak_db,
        "clip_count": clip_count,
    }


def normalize_to_lufs(x: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """将音频归一到目标 LUFS。先去直流，再按 LUFS 缩放。"""
    x = x.astype(np.float32)
    # 去直流
    x = x - np.mean(x)
    meter = pyln.Meter(sr)
    in_lufs = meter.integrated_loudness(x)
    y = pyln.normalize.loudness(x, in_lufs, target_lufs)
    # 可选：避免溢出，限制到 [-1, 1]
    y = np.clip(y, -1.0, 1.0)
    return y


def normalize_original_for_loss_lufs(original_np: np.ndarray,
                                     ref_np: np.ndarray,
                                     sr: int) -> tuple[np.ndarray, float]:
    """
    将原始音频按 LUFS 对齐到参考（重建音频）的 LUFS，用于损失计算。
    返回：归一化后的原始音频，参考的目标 LUFS。
    """
    meter = pyln.Meter(sr)
    target_lufs = float(meter.integrated_loudness(ref_np.astype(np.float32)))
    x_norm = normalize_to_lufs(original_np, sr, target_lufs)
    return x_norm, target_lufs


class MultiResolutionSpectrogramLoss:
    def __init__(self,
                 fft_sizes=[512, 1024, 2048],
                 hop_sizes=[160, 320, 640],
                 win_lengths=[512, 1024, 2048],
                 window='hann_window'):
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.window = window

    def stft(self, x, fft_size, hop_size, win_length):
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
        pred = torch.from_numpy(pred).float() if isinstance(pred, np.ndarray) else pred
        target = torch.from_numpy(target).float() if isinstance(target, np.ndarray) else target

        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)

        total_loss = 0.0
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            pred_spec = self.stft(pred, fft_size, hop_size, win_length)
            target_spec = self.stft(target, fft_size, hop_size, win_length)

            l1_loss = F.l1_loss(pred_spec, target_spec)
            pred_log = torch.log(pred_spec + 1e-7)
            target_log = torch.log(target_spec + 1e-7)
            l2_loss = F.mse_loss(pred_log, target_log)

            total_loss += l1_loss + l2_loss

        return total_loss / len(self.fft_sizes)


# 常见响度标准（简化版检查）
STANDARDS = {
    # EBU R128 广播标准（简化采样峰值检查；严格应使用真峰值 dBTP）
    "ebu_r128": {
        "target_lufs": -23.0,
        "tolerance_lu": 1.0,          # 目标 LUFS ± 容差
        "max_peak_dbfs": -1.0,        # 峰值不高于 -1 dBFS
        "lra_min": 4.0,               # 推荐的最小动态范围（内容依赖）
        "lra_max": 20.0
    },
    # 主流流媒体平台常用 Loudness（范围会因平台/播放端处理不同而不同）
    "streaming": {
        "target_lufs": -14.0,
        "tolerance_lu": 1.0,
        "max_peak_dbfs": -1.0,
        "lra_min": 4.0,
        "lra_max": 20.0
    }
}


def check_standard_compliance(metrics: dict, standard_cfg: dict) -> tuple[bool, list[str]]:
    """
    检查重建音频是否符合某个标准（简化版）。
    返回：是否通过，失败原因列表
    """
    ok = True
    reasons = []
    lufs = metrics["lufs"]
    lra = metrics["lra"]
    peak_dbfs_val = metrics["peak_dbfs"]

    # LUFS 目标 ± 容差
    if not (standard_cfg["target_lufs"] - standard_cfg["tolerance_lu"]
            <= lufs
            <= standard_cfg["target_lufs"] + standard_cfg["tolerance_lu"]):
        ok = False
        reasons.append(f"LUFS={lufs:.2f} 不在目标 {standard_cfg['target_lufs']:.1f}±{standard_cfg['tolerance_lu']:.1f} LU 内")

    # 峰值（采样峰值近似）
    if peak_dbfs_val > standard_cfg["max_peak_dbfs"]:
        ok = False
        reasons.append(f"峰值 {peak_dbfs_val:.2f} dBFS 高于 {standard_cfg['max_peak_dbfs']:.2f} dBFS（可能有削波风险）")

    # 动态范围（LRA）
    if not (standard_cfg["lra_min"] <= lra <= standard_cfg["lra_max"]):
        ok = False
        reasons.append(f"LRA={lra:.2f} 不在 [{standard_cfg['lra_min']:.1f}, {standard_cfg['lra_max']:.1f}] LU 内")

    return ok, reasons


# --- 参数解析 ---
parser = argparse.ArgumentParser(description='从潜在表示还原音频，并进行 LUFS 正则化与合规性检查')
parser.add_argument('--input_latent_path', type=str, required=True,
                    help='输入的 latent npy 文件路径')
parser.add_argument('--output_dir', type=str, default='/blob/avtok/',
                    help='输出目录 (默认: /blob/avtok/)')
parser.add_argument('--device', type=str, default='auto',
                    choices=['cuda', 'cpu', 'auto'],
                    help='运行设备 (默认: auto)')
parser.add_argument('--standard', type=str, default='ebu_r128',
                    choices=list(STANDARDS.keys()),
                    help='响度合规性标准（默认 ebu_r128）')
args = parser.parse_args()

origin_video_base_dir = "/blob/vggsound_cropped/"
latent_base_dir = "/blob/vggsound_cropped_audio_latent_fixed/"
SR = 16000  # 评估与保存使用的采样率（与 pipeline 声码器对齐）

# --- 1. 设置文件路径 ---
input_latent_path = args.input_latent_path
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# --- 2. 检查输入文件是否存在 ---
if not os.path.exists(input_latent_path):
    raise FileNotFoundError(f"错误：找不到 latent 文件 '{input_latent_path}'。请先运行编码脚本。")

# --- 3. 根据 latent 路径构建原始视频路径 ---
relative_path = input_latent_path.replace(latent_base_dir, "")
video_relative_path = relative_path.replace(".npy", ".mp4")
original_video_path = os.path.join(origin_video_base_dir, video_relative_path)

base_name = os.path.splitext(os.path.basename(input_latent_path))[0]
reconstructed_audio_path = os.path.join(output_dir, f"{base_name}_reconstructed.wav")
original_audio_path = os.path.join(output_dir, f"{base_name}_original.wav")

print(f"原始视频路径: {original_video_path}")
print(f"重建音频将保存至: {reconstructed_audio_path}")
print(f"原始音频将保存至: {original_audio_path}")

# --- 4. 从原始视频提取音频并重采样 ---
waveform_original = None
if os.path.exists(original_video_path):
    print("\n正在从视频提取并重采样原始音频...")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
    try:
        cmd = [
            'ffmpeg', '-i', original_video_path,
            '-vn',
            '-ar', str(SR),
            '-ac', '1',
            '-f', 'wav',
            '-y',
            temp_audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            waveform_original, sr = torchaudio.load(temp_audio_path)
            waveform_original = waveform_original.squeeze().numpy().astype(np.float32)
            if sr != SR:
                # 双重保险，一般 ffmpeg 已重采样
                waveform_original = torchaudio.functional.resample(torch.from_numpy(waveform_original), sr, SR).numpy()
            waveform_original = waveform_original[:SR * 3]  # 截取前 3 秒
            scipy.io.wavfile.write(original_audio_path, rate=SR, data=waveform_original)
            print(f"原始音频已提取并重采样，保存至: {original_audio_path}")
            print(f"原始音频形状: {waveform_original.shape}")
        else:
            print(f"警告：无法从视频提取音频。错误信息: {result.stderr}")
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
else:
    print(f"警告：找不到原始视频文件 '{original_video_path}'")

# --- 5. 加载 AudioLDM 2 模型 ---
print("\n正在加载 AudioLDM 2 模型...")
repo_id = "cvssp/audioldm2"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)

# 设置设备
if args.device == 'auto':
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = args.device
pipe = pipe.to(device)
print(f"模型已加载到 {device} 设备。")
vae = pipe.vae
vocoder = pipe.vocoder
print(f"采样率(模型): {pipe.feature_extractor.sampling_rate}")

# --- 6. 加载 Latent 并解码 ---
print(f"\n正在从 '{input_latent_path}' 加载潜在表示...")
latent_np = np.load(input_latent_path)
if latent_np.ndim == 4:
    latent_np = latent_np.squeeze(0)
latent_tensor = torch.from_numpy(latent_np).to(device, dtype=torch.float16).unsqueeze(0)
print(f"Latent 形状: {latent_tensor.shape}")
print("开始解码过程...")

with torch.no_grad():
    print("步骤 1/2: 使用 VAE 解码器将潜在表示转为梅尔频谱图...")
    decoded_mel = vae.decode(latent_tensor).sample
    print("步骤 2/2: 将梅尔频谱图转换为波形...")
    waveform = pipe.mel_spectrogram_to_waveform(decoded_mel)
    waveform = waveform.squeeze().detach().cpu().numpy().astype(np.float32)

print(f"重建音频形状: {waveform.shape}")
waveform = waveform[:SR * 3]  # 截取前 3 秒

# 保存重建音频
scipy.io.wavfile.write(reconstructed_audio_path, rate=SR, data=waveform)
print("\n--- 操作成功 ---")
print(f"重建音频已保存至: '{reconstructed_audio_path}'")
if os.path.exists(original_audio_path):
    print(f"原始音频已保存至: '{original_audio_path}'")

# --- 7. LUFS 正则化 + 指标与合规性检查 ---
if waveform_original is not None:
    print("\n--- 计算损失与响度指标（含 LUFS 正则化） ---")

    # 对齐长度
    min_length = min(len(waveform), len(waveform_original))
    waveform_recon_aligned = waveform[:min_length].astype(np.float32)
    waveform_original_aligned = waveform_original[:min_length].astype(np.float32)

    # LUFS 指标（归一化前）
    orig_metrics_before = compute_loudness_metrics(waveform_original_aligned, SR)
    recon_metrics = compute_loudness_metrics(waveform_recon_aligned, SR)

    # 将原始音频按 LUFS 对齐到重建音频（仅用于损失计算）
    waveform_original_norm, target_lufs = normalize_original_for_loss_lufs(
        waveform_original_aligned, waveform_recon_aligned, SR
    )
    orig_metrics_after = compute_loudness_metrics(waveform_original_norm, SR)

    # 打印 LUFS 与 RMS 以确认缩放情况
    rms_orig_before = np.sqrt(np.mean(waveform_original_aligned ** 2) + 1e-8)
    rms_orig_after = np.sqrt(np.mean(waveform_original_norm ** 2) + 1e-8)
    rms_recon = np.sqrt(np.mean(waveform_recon_aligned ** 2) + 1e-8)
    print(f"LUFS(original_before)={orig_metrics_before['lufs']:.2f}, "
          f"LUFS(original_after)={orig_metrics_after['lufs']:.2f}, "
          f"LUFS(recon)={recon_metrics['lufs']:.2f} (target={target_lufs:.2f})")
    print(f"RMS(original_before)={rms_orig_before:.6f}, RMS(original_after)={rms_orig_after:.6f}, RMS(recon)={rms_recon:.6f}")

    # 1) Waveform L1 / L2 Loss（使用 LUFS 对齐后的原始音频）
    l1_loss = np.mean(np.abs(waveform_recon_aligned - waveform_original_norm))
    l2_loss = np.mean((waveform_recon_aligned - waveform_original_norm) ** 2)
    print(f"Waveform L1 Loss: {l1_loss:.6f}")
    print(f"Waveform L2 Loss (MSE): {l2_loss:.6f}")

    # 2) Multi-Resolution Spectrogram Loss
    print("\n计算 Multi-Resolution Spectrogram Loss...")
    spec_loss_calculator = MultiResolutionSpectrogramLoss()
    waveform_recon_tensor = torch.from_numpy(waveform_recon_aligned).float()
    waveform_original_tensor = torch.from_numpy(waveform_original_norm).float()
    spec_loss = spec_loss_calculator.compute_loss(waveform_recon_tensor, waveform_original_tensor)
    print(f"Multi-Resolution Spectrogram Loss: {spec_loss.item():.6f}")

    # 3) SNR（以 LUFS 对齐后的原始音频作为信号）
    signal_power = float(np.mean(waveform_original_norm ** 2))
    noise_power = float(np.mean((waveform_recon_aligned - waveform_original_norm) ** 2))
    snr = 10.0 * np.log10(signal_power / (noise_power + 1e-12))
    print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB")

    # 4) 相关系数
    correlation = float(np.corrcoef(waveform_recon_aligned, waveform_original_norm)[0, 1])
    print(f"Correlation Coefficient: {correlation:.4f}")

    # 5) 合规性检查（对重建音频）
    standard_cfg = STANDARDS[args.standard]
    recon_ok, reasons = check_standard_compliance(recon_metrics, standard_cfg)
    print(f"\n合规性检查标准: {args.standard}")
    print(f"- 目标 LUFS: {standard_cfg['target_lufs']} ± {standard_cfg['tolerance_lu']} LU")
    print(f"- 峰值上限: {standard_cfg['max_peak_dbfs']} dBFS（采样峰值近似）")
    print(f"- 动态范围(LRA) 推荐区间: [{standard_cfg['lra_min']}, {standard_cfg['lra_max']}] LU")
    print(f"重建音频合规: {'通过' if recon_ok else '未通过'}")
    if not recon_ok:
        for r in reasons:
            print(f"  不通过原因: {r}")

    # 保存指标到文件
    metrics_path = os.path.join(output_dir, f"{base_name}_metrics.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"音频重建质量评估指标（含 LUFS 正则化与合规性检查）\n")
        f.write("=" * 60 + "\n")
        f.write(f"文件: {base_name}\n")
        f.write(f"采样率: {SR} Hz\n")
        f.write(f"长度(对齐后): {min_length} samples\n\n")

        f.write("响度指标（归一化前/后 vs 重建）\n")
        f.write(f"- Original (before) LUFS: {orig_metrics_before['lufs']:.2f}, "
                f"LRA: {orig_metrics_before['lra']:.2f}, "
                f"Peak: {orig_metrics_before['peak_dbfs']:.2f} dBFS, "
                f"Clips: {orig_metrics_before['clip_count']}\n")
        f.write(f"- Original (after LUFS align to recon={target_lufs:.2f}) LUFS: {orig_metrics_after['lufs']:.2f}, "
                f"LRA: {orig_metrics_after['lra']:.2f}, "
                f"Peak: {orig_metrics_after['peak_dbfs']:.2f} dBFS, "
                f"Clips: {orig_metrics_after['clip_count']}\n")
        f.write(f"- Reconstructed LUFS: {recon_metrics['lufs']:.2f}, "
                f"LRA: {recon_metrics['lra']:.2f}, "
                f"Peak: {recon_metrics['peak_dbfs']:.2f} dBFS, "
                f"Clips: {recon_metrics['clip_count']}\n\n")

        f.write("损失指标（以 LUFS 对齐后的原始音频作为参考）\n")
        f.write(f"- Waveform L1 Loss: {l1_loss:.6f}\n")
        f.write(f"- Waveform L2 Loss (MSE): {l2_loss:.6f}\n")
        f.write(f"- Multi-Resolution Spectrogram Loss: {spec_loss.item():.6f}\n\n")

        f.write("质量指标\n")
        f.write(f"- SNR: {snr:.2f} dB\n")
        f.write(f"- Correlation: {correlation:.4f}\n\n")

        f.write(f"合规性检查标准: {args.standard}\n")
        f.write(f"- 目标 LUFS: {standard_cfg['target_lufs']} ± {standard_cfg['tolerance_lu']} LU\n")
        f.write(f"- 峰值上限: {standard_cfg['max_peak_dbfs']} dBFS（采样峰值近似）\n")
        f.write(f"- 动态范围(LRA) 推荐区间: [{standard_cfg['lra_min']}, {standard_cfg['lra_max']}] LU\n")
        f.write(f"- 结果: {'通过' if recon_ok else '未通过'}\n")
        if not recon_ok:
            for r in reasons:
                f.write(f"  不通过原因: {r}\n")

    print(f"\n指标已保存至: {metrics_path}")
else:
    print("\n警告：无法计算损失与响度指标，因为原始音频不可用。")

print(f"\n所有文件都保存在: '{output_dir}'")
