import sys
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from typing import Dict, Union, List, Optional
import soundfile as sf
from pathlib import Path
import os
from train import AudioVAEFSQLightningModule

class MultiResolutionSTFTLoss(nn.Module):
    """改进的多分辨率STFT损失，参考AudioLDM2"""
    def __init__(self,
                 fft_sizes: List[int] = [512, 1024, 2048],      # AudioLDM2风格
                 hop_sizes: List[int] = [160, 320, 640],        # 10ms, 20ms, 40ms
                 win_lengths: List[int] = [512, 1024, 2048],
                 mag_weight: float = 1.0,
                 log_mag_weight: float = 1.0,
                 sample_rate: int = 16000):
        super().__init__()
        
        self.stft_losses = nn.ModuleList()
        self.mag_weight = mag_weight
        self.log_mag_weight = log_mag_weight
        
        for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths):
            win = min(win, n_fft)
            self.stft_losses.append(
                T.Spectrogram(
                    n_fft=n_fft,
                    hop_length=hop,
                    win_length=win,
                    power=None,
                    normalized=True,
                    pad_mode='reflect',
                    center=True
                )
            )
    
    def forward(self, pred_waveform: torch.Tensor, true_waveform: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        
        for stft in self.stft_losses:
            stft = stft.to(pred_waveform.device)
            
            pred_stft = stft(pred_waveform)
            true_stft = stft(true_waveform)
            
            pred_mag = torch.abs(pred_stft)
            true_mag = torch.abs(true_stft)
            
            mag_loss = F.l1_loss(pred_mag, true_mag)
            
            eps = 1e-7
            pred_log_mag = torch.log(pred_mag.clamp(min=eps))
            true_log_mag = torch.log(true_mag.clamp(min=eps))
            log_mag_loss = F.l1_loss(pred_log_mag, true_log_mag)
            
            loss += self.mag_weight * mag_loss + self.log_mag_weight * log_mag_loss
            
        return loss / len(self.stft_losses)


class MelSpectrogramLoss(nn.Module):
    """梅尔频谱感知损失"""
    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 1024,
                 hop_length: int = 160,  # AudioLDM2: 10ms
                 n_mels: int = 64,       # AudioLDM2默认
                 fmin: float = 0.0,
                 fmax: float = 8000.0):
        super().__init__()
        
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            normalized=True,
            pad_mode='reflect',
            center=True
        )
        
    def forward(self, pred_waveform: torch.Tensor, true_waveform: torch.Tensor) -> torch.Tensor:
        self.mel_spec = self.mel_spec.to(pred_waveform.device)
        
        pred_mel = self.mel_spec(pred_waveform)
        true_mel = self.mel_spec(true_waveform)
        
        eps = 1e-7
        pred_log_mel = torch.log(pred_mel.clamp(min=eps))
        true_log_mel = torch.log(true_mel.clamp(min=eps))
        
        loss = F.l1_loss(pred_log_mel, true_log_mel)
        
        return loss

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
        output = model(waveform, return_dict=True,duration=3)
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
    stft_loss_fn=MultiResolutionSTFTLoss()
    mel_loss_fn=MelSpectrogramLoss()
    stft_loss=stft_loss_fn(reconstructed,waveform)
    mel_loss=mel_loss_fn(reconstructed,waveform)

    print("MultiResolutionSTFTLoss: ",stft_loss)
    print("MelSpectrogramLoss: ",mel_loss)
    
    return recon_path


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python simple_inference.py <audio_file> <checkpoint> <output_dir>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    checkpoint = sys.argv[2]
    output_dir = sys.argv[3]
    
    reconstruct_audio(audio_file, checkpoint, output_dir)
