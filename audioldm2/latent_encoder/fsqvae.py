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
        
        # 数值稳定性参数
        self.eps = 1e-8
        self.max_val = 50.0
        
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

    def _stabilize_tensor(self, x: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """数值稳定化处理"""
        # 检查NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: NaN/Inf detected in {name}, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0, posinf=self.max_val, neginf=-self.max_val)
        
        # 限制范围
        x = torch.clamp(x, -self.max_val, self.max_val)
        return x

    def _encode_no_quant(self, x: torch.Tensor) -> torch.Tensor:
        """内部编码方法，不进行量化"""
        # 稳定化输入
        x = self._stabilize_tensor(x, "encoder_input")
        
        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            enc = self._tiled_encode_no_quant(x)
        else:
            enc = self.encoder(x)
        
        # 稳定化输出
        enc = self._stabilize_tensor(enc, "encoder_output")
        return enc
    
    def _tiled_encode_no_quant(self, x: torch.Tensor) -> torch.Tensor:
        """Tiled编码，不进行量化"""
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent
        
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                tile = self._stabilize_tensor(tile, "tile_encode")
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
               duration: float = 1.1) -> Union[dict, Tuple]:
        """
        编码流程：音频预处理 -> VAE编码 -> FSQ量化
        """
        # 1. 音频预处理
        if hasattr(self, 'fn_STFT') and x.dim() in [1, 2, 3]:
            # 归一化音频
            x = self._stabilize_tensor(x, "audio_input")
            if x.abs().max() > 1.0:
                x = x / (x.abs().max() + self.eps)
            
            fbank, _, _ = self.wav_to_fbank_batch(
                batch_waveforms=x, 
                target_length=int(duration * 102.4), 
                fn_STFT=self.fn_STFT
            )
            mel_spectrogram = fbank.unsqueeze(1)
            mel_spectrogram = self._stabilize_tensor(mel_spectrogram, "mel_spectrogram")
        else:
            mel_spectrogram = self._stabilize_tensor(x, "input")
        
        # 2. VAE编码
        if self.use_slicing and mel_spectrogram.shape[0] > 1:
            encoded_slices = [self._encode_no_quant(x_slice) for x_slice in mel_spectrogram.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode_no_quant(mel_spectrogram)
        
        # 3. 创建稳定的分布
        # 分离均值和对数方差
        mean, logvar = torch.chunk(h, 2, dim=1)
        
        # 限制logvar范围
        logvar = torch.clamp(logvar, -30.0, 20.0)
        
        # 重新组合
        h_stable = torch.cat([mean, logvar], dim=1)
        posterior = DiagonalGaussianDistribution(h_stable)
        
        # 4. 获取latent（训练时采样，推理时用mode）
        if self.training:
            z = posterior.sample()
        else:
            z = posterior.mode()
        
        z = self._stabilize_tensor(z, "latent")
        
        # 5. FSQ量化
        # 自适应温度调度
        if n_steps < 1000:
            inv_temperature = 0.1
        elif n_steps < 5000:
            inv_temperature = 0.5
        else:
            inv_temperature = min(1.0, 0.5 + (n_steps - 5000) / 10000)
        
        # 在float32精度下进行量化以提高稳定性
        with torch.cuda.amp.autocast(enabled=False):
            z_float = z.float()
            try:
                z_quantized, fsq_dict = self.quantizer(z_float, n_steps=n_steps, inv_temperature=inv_temperature)
                z_quantized = z_quantized.to(z.dtype)
            except Exception as e:
                print(f"FSQ quantization error: {e}, using bypass")
                z_quantized = z
                fsq_dict = {"aux_loss": torch.tensor(0.0).to(z.device)}
        
        z_quantized = self._stabilize_tensor(z_quantized, "quantized_latent")
        
        # 确保fsq_dict包含必要的信息
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

    def _decode_no_post_quant(self, z: torch.Tensor) -> torch.Tensor:
        """内部解码方法"""
        z = self._stabilize_tensor(z, "decoder_input")
        
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            dec = self._tiled_decode_no_post_quant(z)
        else:
            dec = self.decoder(z)
        
        dec = self._stabilize_tensor(dec, "decoder_output")
        return dec
    
    def _tiled_decode_no_post_quant(self, z: torch.Tensor) -> torch.Tensor:
        """Tiled解码"""
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent
        
        rows = []
        for i in range(0, z.shape[2], overlap_size):
            row = []
            for j in range(0, z.shape[3], overlap_size):
                tile = z[:, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                decoded = self.decoder(tile)
                decoded = self._stabilize_tensor(decoded, "tile_decode")
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
        """解码流程"""
        # VAE解码
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode_no_post_quant(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode_no_post_quant(z)
        
        # 转换为音频（如果需要）
        if to_waveform and hasattr(self, 'vocoder'):
            decoded = self.mel_spectrogram_to_waveform(decoded)
            decoded = self._stabilize_tensor(decoded, "waveform_output")
        
        if not return_dict:
            return (decoded,)
        
        return DecoderOutput(sample=decoded)

    def forward(self, sample: torch.Tensor, n_steps: int = 0, return_dict: bool = True,
                duration: float = 1.1):
        """
        前向传播：编码 -> 量化 -> 解码
        返回重建结果和FSQ字典供外部计算损失
        """
        # 编码和量化
        encode_outputs = self.encode(
            sample, 
            return_dict=True, 
            n_steps=n_steps,
            duration=duration
        )
        
        z_quantized = encode_outputs["quantized_latent"]
        fsq_dict = encode_outputs["fsq_dict"]
        
        # 解码为波形
        reconstruction = self.decode(z_quantized, return_dict=True, to_waveform=True).sample
        
        if not return_dict:
            return reconstruction, fsq_dict
        
        return {
            "reconstruction": reconstruction,
            "quantized_latent": z_quantized,
            "unquantized_latent": encode_outputs["unquantized_latent"],
            "posterior": encode_outputs["posterior"],
            "fsq_dict": fsq_dict
        }

    # 音频处理辅助方法保持不变
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