import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional
from diffusers import AutoencoderKL
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DecoderOutput,DiagonalGaussianDistribution


class AutoencoderFSQ(AutoencoderKL):
    def __init__(
        self,
        # AutoencoderKL参数
        in_channels: int = 1,
        out_channels: int = 1,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels: Tuple[int] = (128, 256, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 8,
        norm_num_groups: int = 32,
        sample_size: int = 1024,
        scaling_factor: float = 0.4110932946205139,
        
        # FSQ参数
        fsq_levels: list = [8, 8, 8, 8, 8],
        fsq_commitment_loss_weight: float = 0.25,
        sampling_rate: int = 16000,
        target_mel_length: int = 1024,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            use_quant_conv=False,  # 禁用原始量化层
            use_post_quant_conv=False,
        )
        
        # 初始化FSQ
        from audioldm2.modules.regularizers import FSQRegularizer
        self.quantizer = FSQRegularizer(
            levels=fsq_levels,
            dim=latent_channels,
            use_projection=True,
            commitment_loss_weight=fsq_commitment_loss_weight,
            entropy_loss_weight=0.0,
        )
        
        # 音频处理参数
        self.sampling_rate = sampling_rate
        self.target_mel_length = target_mel_length
        
        # 初始化音频预处理
        from audioldm2.utils import default_audioldm_config
        from audioldm2.utilities.audio.stft import TacotronSTFT
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
        
        # 初始化vocoder
        from transformers import SpeechT5HifiGan
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "cvssp/audioldm2",
            subfolder="vocoder",
            torch_dtype=torch.float32
        )

    def _encode_no_quant(self, x: torch.Tensor) -> torch.Tensor:
        """
        内部编码方法，不进行量化，直接使用encoder
        """
        batch_size, num_channels, height, width = x.shape
        
        # 处理tiling情况
        if self.use_tiling and (width > self.tile_sample_min_size or height > self.tile_sample_min_size):
            return self._tiled_encode_no_quant(x)
        
        # 直接使用encoder，不使用quant_conv
        enc = self.encoder(x)
        return enc
    
    def _tiled_encode_no_quant(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tiled编码，不进行量化
        """
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent
        
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)
            
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))
        
        enc = torch.cat(result_rows, dim=2)
        return enc

    def encode(self, x: torch.Tensor, return_dict: bool = True, n_steps: int = 0, 
               duration: float = 1.1, inv_temperature: float = 0.1) -> Union[dict, Tuple]:
        """
        完整的编码流程：音频预处理 -> VAE编码 -> FSQ量化
        """
        # 1. 音频预处理（如果输入是音频）
        if hasattr(self, 'fn_STFT') and x.dim() in [1, 2, 3]:
            # 处理音频输入
            fbank, _, _ = self.wav_to_fbank_batch(
                batch_waveforms=x, 
                target_length=int(duration * 102.4), 
                fn_STFT=self.fn_STFT
            )
            mel_spectrogram = fbank.unsqueeze(1)
        else:
            # 假设输入已经是mel谱图
            mel_spectrogram = x
        
        # 2. VAE编码（不使用super）
        if self.use_slicing and mel_spectrogram.shape[0] > 1:
            encoded_slices = [self._encode_no_quant(x_slice) for x_slice in mel_spectrogram.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode_no_quant(mel_spectrogram)
        
        # 3. 创建分布并采样
        posterior = DiagonalGaussianDistribution(h)
        z = posterior.mode()  # 或使用 posterior.sample()
        
        # 4. FSQ量化
        z_quantized, fsq_dict = self.quantizer(z, n_steps=n_steps, inv_temperature=inv_temperature)
        
        # 5. 添加额外信息
        fsq_dict['posterior'] = posterior  # 保存原始分布用于KL loss
        fsq_dict['unquantized'] = z  # 保存未量化的latent
        
        if not return_dict:
            return (z_quantized, fsq_dict)
        
        return {
            "quantized_latent": z_quantized,
            "fsq_dict": fsq_dict,
            "posterior": posterior,
            "unquantized_latent": z
        }

    def _decode_no_post_quant(self, z: torch.Tensor) -> torch.Tensor:
        """
        内部解码方法，不使用post_quant_conv
        """
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self._tiled_decode_no_post_quant(z)
        
        # 直接使用decoder，不使用post_quant_conv
        dec = self.decoder(z)
        return dec
    
    def _tiled_decode_no_post_quant(self, z: torch.Tensor) -> torch.Tensor:
        """
        Tiled解码，不使用post_quant_conv
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent
        
        rows = []
        for i in range(0, z.shape[2], overlap_size):
            row = []
            for j in range(0, z.shape[3], overlap_size):
                tile = z[:, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
            
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))
        
        dec = torch.cat(result_rows, dim=2)
        return dec

    def decode(self, z: torch.FloatTensor, return_dict: bool = True, 
               generator=None, to_waveform: bool = False) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        完整的解码流程：FSQ latents -> VAE解码 -> (可选)转换为音频
        """
        # 1. VAE解码（不使用super）
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode_no_post_quant(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode_no_post_quant(z)
        
        # 2. 如果需要，转换为音频波形
        if to_waveform and hasattr(self, 'vocoder'):
            decoded = self.mel_spectrogram_to_waveform(decoded)
        
        if not return_dict:
            return (decoded,)
        
        return DecoderOutput(sample=decoded)

    def forward(self, sample: torch.Tensor, n_steps: int = 0, return_dict: bool = True,
                sample_posterior: bool = False, inv_temperature: float = 1.0, 
                to_waveform: bool = True, duration: float = 1.1):
        """
        完整的前向传播：编码 -> FSQ量化 -> 解码
        
        Args:
            sample: 输入音频或mel谱图
            n_steps: 训练步数（用于FSQ）
            return_dict: 是否返回字典
            sample_posterior: 是否从分布中采样（而不是使用mode）
            inv_temperature: FSQ的温度参数
            to_waveform: 是否将输出转换为音频波形
            duration: 音频时长
        """
        # 1. 编码和量化
        encode_outputs = self.encode(
            sample, 
            return_dict=True, 
            n_steps=n_steps,
            duration=duration,
            inv_temperature=inv_temperature
        )
        
        z_quantized = encode_outputs["quantized_latent"]
        fsq_dict = encode_outputs["fsq_dict"]
        posterior = encode_outputs["posterior"]
        z_unquantized = encode_outputs["unquantized_latent"]
        
        # 2. 解码
        reconstruction = self.decode(z_quantized, return_dict=True, to_waveform=to_waveform).sample
        
        # 3. 计算损失
        if self.training:
            # 重建损失
            if to_waveform:
                rec_loss = F.mse_loss(reconstruction, sample)
            else:
                # 如果是mel谱图，需要先转换
                target_mel = self.wav_to_fbank_batch(sample, int(duration * 102.4), self.fn_STFT)[0]
                rec_loss = F.mse_loss(reconstruction, target_mel.unsqueeze(1))
            
            # KL损失
            kl_loss = posterior.kl().mean()
            
            # FSQ相关损失（commitment loss等）
            fsq_loss = fsq_dict.get('loss', 0.0)
            
            total_loss = rec_loss + 0.1 * kl_loss + fsq_loss
        else:
            total_loss = None
        
        if not return_dict:
            return reconstruction, fsq_dict, total_loss
        
        return {
            "reconstruction": reconstruction,
            "quantized_latent": z_quantized,
            "unquantized_latent": z_unquantized,
            "posterior": posterior,
            "fsq_dict": fsq_dict,
            "loss": total_loss,
            "rec_loss": rec_loss if self.training else None,
            "kl_loss": kl_loss if self.training else None,
        }

    # 保留音频处理的辅助方法
    def wav_to_fbank_batch(self, batch_waveforms, target_length=1024, fn_STFT=None, device=None):
        """音频转mel谱图"""
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

    def get_mel_from_wav_batch(self, audio_batch, _stft):
        """获取mel谱图"""
        if audio_batch.dim() == 3:
            audio_batch = audio_batch.squeeze(1)
        elif audio_batch.dim() == 1:
            audio_batch = audio_batch.unsqueeze(0)
        
        assert audio_batch.dim() == 2, f"Expected 2D tensor, got {audio_batch.dim()}D"
        
        audio_batch = torch.clip(audio_batch, -1, 1)
        audio_batch = torch.autograd.Variable(audio_batch, requires_grad=False)
        
        melspec, magnitudes, phases, energy = _stft.mel_spectrogram(audio_batch)
        return melspec, magnitudes, energy

    def _pad_spec(self, fbank, target_length=1024):
        """填充谱图"""
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

    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        """mel谱图转音频"""
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)
        waveform = self.vocoder(mel_spectrogram)
        return waveform
