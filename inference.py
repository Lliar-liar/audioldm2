import sys
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
import os
from train import AudioVAEFSQLightningModule


def reconstruct_audio(audio_path: str, checkpoint_path: str, output_dir: str, device: str = 'cuda'):
    """重建音频文件"""
    
    # 加载模型
    print(f"Loading model from {checkpoint_path}")
    model = AudioVAEFSQLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    ).model
    model.eval()
    model.to(device)
    
    # 加载音频
    print(f"Loading audio from {audio_path}")
    waveform, sr = torchaudio.load(audio_path)
    
    # 重采样到16kHz（如果需要）
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    
    # 转单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # 推理
    print("Reconstructing...")
    with torch.no_grad():
        waveform = waveform.unsqueeze(0).to(device)  # 添加batch维度
        output = model(waveform, return_dict=True)
        reconstructed = output['reconstruction'].cpu()[0]  # 移除batch维度
    
    # 保存结果
    input_path = Path(audio_path)
    output_dir = output_dir
    
    # 保存重建音频

    recon_path = os.path.join(output_dir , f"{input_path.stem}_reconstructed.wav")
    sf.write(recon_path, reconstructed.numpy().T, 16000)
    print(f"✓ Saved: {recon_path}")
    
    # 保存原始音频（处理后）
    original_path = os.path.join(output_dir , f"{input_path.stem}_original_16k.wav")
    sf.write(original_path, waveform.cpu()[0].numpy().T, 16000)
    print(f"✓ Saved: {original_path}")
    
    # 计算简单指标
    mse = torch.mean((waveform.cpu() - reconstructed.unsqueeze(0)) ** 2).item()
    print(f"\nMSE: {mse:.6f}")
    
    return str(recon_path)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python simple_inference.py <audio_file> <checkpoint> <output_dir>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    checkpoint = sys.argv[2]
    output_dir = sys.argv[3]
    
    reconstruct_audio(audio_file, checkpoint, output_dir)
