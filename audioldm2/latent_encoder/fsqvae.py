import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional, Dict, Any
from diffusers import AutoencoderKL
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution, DecoderOutput
from transformers import SpeechT5HifiGan
from audioldm2.modules.regularizers import FSQRegularizer
from audioldm2.utils import default_audioldm_config
from audioldm2.utilities.audio.stft import TacotronSTFT


class AutoencoderFSQ(nn.Module):
    """
    VAE with FSQ quantization wrapper
    使用预训练的AudioLDM2 VAE作为基础，添加FSQ量化
    """
    
    def __init__(
        self,
        model_name: str = "cvssp/audioldm2",
        subfolder: str = "vae",
        fsq_levels: list = [8, 8, 8, 8, 8],
        fsq_commitment_loss_weight: float = 0.25,
        sampling_rate: int = 16000,
        target_mel_length: int = 1024,
        torch_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        # 1. 加载预训练的VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_name,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
        )
        
        # 获取VAE的latent维度
        self.latent_channels = self.vae.config.latent_channels
        
        # 2. 初始化FSQ量化器
        self.quantizer = FSQRegularizer(
            levels=fsq_levels,
            dim=self.latent_channels,
            use_projection=True,
            commitment_loss_weight=fsq_commitment_loss_weight,
            entropy_loss_weight=0.0,
        )
        
        # 3. 音频处理参数
        self.sampling_rate = sampling_rate
        self.target_mel_length = target_mel_length
        
        # 4. 初始化音频预处理模块
        config_audio = default_audioldm_config()
        self.fn_STFT = TacotronSTFT(
            config_audio["preprocessing"]["stft"]["filter_length"],
            config_audio["preprocessing"]["stft"]["hop_length"],
            config_audio["preprocessing"]["stft"]["win_length"],
            config_audio["preprocessing"]["mel"]["n_mel_channels"],
            self.sampling_rate,
            config_audio["preprocessing"]["mel"]["mel_fmin"],
            config_audio["preprocessing"]["mel"]["mel_fmax"],
        )
        
        # 5. 初始化vocoder
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            model_name,
            subfolder="vocoder",
            torch_dtype=torch_dtype
        )
        
        # 数值稳定性参数
        self.eps = 1e-8
        self.max_val = 50.0
        
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """便捷的类方法用于加载预训练模型"""
        return cls(model_name=model_name, **kwargs)
    
    def _stabilize_tensor(self, x: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """数值稳定化处理"""
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: NaN/Inf detected in {name}, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0, posinf=self.max_val, neginf=-self.max_val)
        x = torch.clamp(x, -self.max_val, self.max_val)
        return x
    
    def mel_spectrogram_to_waveform(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """Mel谱图转波形"""
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)
        waveform = self.vocoder(mel_spectrogram)
        return waveform
    
    def get_mel_from_wav_batch(self, audio_batch: torch.Tensor, _stft) -> Tuple:
        """批量音频转Mel谱图"""
        if audio_batch.dim() == 3:
            audio_batch = audio_batch.squeeze(1)
        elif audio_batch.dim() == 1:
            audio_batch = audio_batch.unsqueeze(0)
        
        assert audio_batch.dim() == 2, f"Expected 2D tensor, got {audio_batch.dim()}D"
        
        audio_batch = torch.clip(audio_batch, -1, 1)
        audio_batch = torch.autograd.Variable(audio_batch, requires_grad=False)
        
        melspec, magnitudes, phases, energy = _stft.mel_spectrogram(audio_batch)
        return melspec, magnitudes, energy
    
    def _pad_spec(self, fbank: torch.Tensor, target_length: int = 1024) -> torch.Tensor:
        """填充或裁剪频谱图"""
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
        
        if fbank.size(-1) % 2 != 0:
            fbank = fbank[..., :-1]
        
        return fbank
    
    def wav_to_fbank_batch(self, batch_waveforms: torch.Tensor, target_length: int = 1024, 
                          fn_STFT=None, device=None) -> Tuple:
        """批量波形转Fbank特征"""
        assert fn_STFT is not None
        
        fbank_batch, log_magnitudes_stft_batch, energy_batch = self.get_mel_from_wav_batch(
            batch_waveforms, fn_STFT
        )
        
        processed_fbanks = []
        for i in range(fbank_batch.size(0)):
            fbank = fbank_batch[i].T
            fbank = self._pad_spec(fbank, target_length)
            processed_fbanks.append(fbank)
        
        fbank_batch = torch.stack(processed_fbanks)
        return fbank_batch, log_magnitudes_stft_batch, batch_waveforms
    
    def encode(self, x: torch.Tensor, return_dict: bool = True, n_steps: int = 0, 
               duration: float = 1.1) -> Union[Dict, Tuple]:
        """
        编码音频为量化的latent表示
        
        Args:
            x: 输入音频波形 [B, T] 或 [B, 1, T]
            return_dict: 是否返回字典
            n_steps: 当前训练步数（用于FSQ温度调度）
            duration: 音频时长（秒）
        
        Returns:
            字典包含：quantized_latent, fsq_dict, posterior, unquantized_latent
        """
        # 1. 音频预处理 - 转换为Mel谱图
        x = self._stabilize_tensor(x, "audio_input")
        
        # 归一化音频
        if x.abs().max() > 1.0:
            x = x / (x.abs().max() + self.eps)
        
        fbank, _, _ = self.wav_to_fbank_batch(
            batch_waveforms=x, 
            target_length=int(duration * 102.4), 
            fn_STFT=self.fn_STFT
        )
        mel_spectrogram = fbank.unsqueeze(1)
        mel_spectrogram = self._stabilize_tensor(mel_spectrogram, "mel_spectrogram")
        
        # 2. 使用VAE编码
        with torch.no_grad() if not self.training else torch.enable_grad():
            vae_outputs = self.vae.encode(mel_spectrogram, return_dict=True)
        
        posterior = vae_outputs.latent_dist
        
        # 3. 从分布采样或取mode
        if self.training:
            z = posterior.sample()
        else:
            z = posterior.mode()
        
        z = self._stabilize_tensor(z, "latent")
        
        # 4. FSQ量化（带温度调度）
        if n_steps < 1000:
            inv_temperature = 0.1
        elif n_steps < 5000:
            inv_temperature = 0.5
        else:
            inv_temperature = min(1.0, 0.5 + (n_steps - 5000) / 10000)
        
        # 在float32下进行量化以提高稳定性
        with torch.cuda.amp.autocast(enabled=False):
            z_float = z.float()
            try:
                z_quantized, fsq_dict = self.quantizer(
                    z_float, 
                    n_steps=n_steps, 
                    inv_temperature=inv_temperature
                )
                z_quantized = z_quantized.to(z.dtype)
            except Exception as e:
                print(f"FSQ quantization error: {e}, using bypass")
                z_quantized = z
                fsq_dict = {"aux_loss": torch.tensor(0.0).to(z.device)}
        
        z_quantized = self._stabilize_tensor(z_quantized, "quantized_latent")
        
        # 添加额外信息
        fsq_dict['posterior'] = posterior
        fsq_dict['unquantized'] = z
        
        if not return_dict:
            return (z_quantized, fsq_dict)
        
        return {
            "quantized_latent": z_quantized,
            "fsq_dict": fsq_dict,
            "posterior": posterior,
            "unquantized_latent": z
        }
    
    def decode(self, z: torch.Tensor, return_dict: bool = True, 
               to_waveform: bool = False, **kwargs) -> Union[DecoderOutput, torch.Tensor]:
        """
        解码latent表示为Mel谱图或波形
        
        Args:
            z: Latent表示
            return_dict: 是否返回字典
            to_waveform: 是否转换为音频波形
        
        Returns:
            解码后的Mel谱图或波形
        """
        # 稳定化输入
        z = self._stabilize_tensor(z, "decoder_input")
        
        # 使用VAE解码
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.vae.decode(z, return_dict=True)
        
        decoded = outputs.sample
        decoded = self._stabilize_tensor(decoded, "decoded_mel")
        
        # 如果需要，转换为波形
        if to_waveform:
            decoded = self.mel_spectrogram_to_waveform(decoded)
            decoded = self._stabilize_tensor(decoded, "waveform")
        
        if not return_dict:
            return (decoded,)
        
        return DecoderOutput(sample=decoded)
    
    def forward(self, sample: torch.Tensor, n_steps: int = 0, return_dict: bool = True,
                duration: float = 1.1) -> Union[Dict, Tuple]:
        """
        完整的前向传播：编码 -> 量化 -> 解码
        
        Args:
            sample: 输入音频
            n_steps: 训练步数
            return_dict: 是否返回字典
            duration: 音频时长
        
        Returns:
            重建的音频和相关信息
        """
        # 1. 编码和量化
        encode_outputs = self.encode(
            sample, 
            return_dict=True, 
            n_steps=n_steps,
            duration=duration
        )
        
        z_quantized = encode_outputs["quantized_latent"]
        fsq_dict = encode_outputs["fsq_dict"]
        posterior = encode_outputs["posterior"]
        z_unquantized = encode_outputs["unquantized_latent"]
        
        # 2. 解码为波形
        reconstruction = self.decode(z_quantized, return_dict=True, to_waveform=True).sample
        
        if not return_dict:
            return (reconstruction, fsq_dict)
        
        return {
            "reconstruction": reconstruction,
            "quantized_latent": z_quantized,
            "unquantized_latent": z_unquantized,
            "posterior": posterior,
            "fsq_dict": fsq_dict
        }
    
    def train(self, mode: bool = True):
        """设置训练模式"""
        super().train(mode)
        self.vae.train(mode)
        return self
    
    def eval(self):
        """设置评估模式"""
        super().eval()
        self.vae.eval()
        return self
    
    @property
    def device(self):
        """获取设备"""
        return next(self.parameters()).device
    
    def to(self, *args, **kwargs):
        """移动到设备"""
        super().to(*args, **kwargs)
        self.vae = self.vae.to(*args, **kwargs)
        return self
