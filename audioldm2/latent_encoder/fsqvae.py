import torch
import os
import sys
import torch.nn.functional as F
import numpy as np
from audioldm2.latent_diffusion.modules.ema import *

from audioldm2.latent_diffusion.modules.diffusionmodules.model import Encoder, Decoder
from audioldm2.latent_diffusion.modules.distributions.distributions import (
    DiagonalGaussianDistribution,
)
import soundfile as sf

from audioldm2.utilities.model import get_vocoder
from audioldm2.utilities.tools import synth_one_sample
from typing import Union, Tuple


from diffusers import AutoencoderKL
from transformers import SpeechT5HifiGan
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from audioldm2.modules.regularizers import FSQRegularizer

from audioldm2.utils import default_audioldm_config
from audioldm2.utilities.audio.stft import TacotronSTFT

def _pad_spec(fbank, target_length=1024):
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





class AutoencoderFSQ(AutoencoderKL):

    def __init__(
        self,
        # --- 继承自 AutoencoderKL 的参数 ---
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
        
        fsq_levels: list = [8, 8, 8, 8, 8],
        fsq_commitment_loss_weight: float = 0.25,
        sampling_rate: int = 16000,
        target_mel_length: int = 1024,
    ):
 

        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            down_block_types=down_block_types, up_block_types=up_block_types,
            block_out_channels=block_out_channels, layers_per_block=layers_per_block,
            act_fn=act_fn, latent_channels=latent_channels,
            norm_num_groups=norm_num_groups, sample_size=sample_size,
            scaling_factor=scaling_factor,
            use_quant_conv=True,
            use_post_quant_conv=True,
        )

        # --- 1. 初始化 FSQ Regularizer ---
        # FSQ 的维度必须等于 VAE 的 latent_channels
        # assert latent_channels == len(fsq_levels), (
        #     f"'latent_channels' ({latent_channels}) must match the number of FSQ levels ({len(fsq_levels)})."
        # )
        # print(latent_channels)
        self.quantizer = FSQRegularizer(
            levels=fsq_levels,
            dim=latent_channels,
            use_projection=True, # FSQ 直接处理 VAE Encoder 的输出
            # commitment_loss_weight=fsq_commitment_loss_weight,
            commitment_loss_weight=1e-6,
            entropy_loss_weight=0.0,
        )
        
        # --- 2. 初始化音频预处理模块 ---
        self.sampling_rate = sampling_rate
        self.target_mel_length = target_mel_length
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
        self.vocoder=SpeechT5HifiGan.from_pretrained(
            "cvssp/audioldm2",
            subfolder="vocoder",
            torch_dtype=torch.float32
        )

    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform
        return waveform

    def get_mel_from_wav_batch(self, audio_batch, _stft):
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
        

    
    def wav_to_fbank_batch(self, batch_waveforms, target_length=1024, fn_STFT=None, device=None):

        assert fn_STFT is not None

        fbank_batch, log_magnitudes_stft_batch, energy_batch = self.get_mel_from_wav_batch(
            batch_waveforms, fn_STFT
        )
        
        processed_fbanks = []
        processed_log_mags = []
        
        for i in range(fbank_batch.size(0)):
            fbank = fbank_batch[i].T
            log_mag = log_magnitudes_stft_batch[i].T
            
            fbank = _pad_spec(fbank, target_length)
            log_mag = _pad_spec(log_mag, target_length)
            
            processed_fbanks.append(fbank)
            processed_log_mags.append(log_mag)
        
        fbank_batch = torch.stack(processed_fbanks)
        log_magnitudes_stft_batch = torch.stack(processed_log_mags)
        
        return fbank_batch, log_magnitudes_stft_batch, batch_waveforms

    def encode(self, x: torch.Tensor, return_dict: bool = True, n_steps: int = 0, duration :float=1.1) -> Union[AutoencoderKLOutput, Tuple]:

        
        fbank, _, _, = self.wav_to_fbank_batch(batch_waveforms=x, target_length=int(duration * 102.4), fn_STFT=self.fn_STFT)
 
        mel_spectrogram = fbank.unsqueeze(1)
        # print(mel_spectrogram.shape)
        # print(torch.isfinite(mel_spectrogram))
        posterior_output = super().encode(mel_spectrogram, return_dict=True)
        posterior = posterior_output.latent_dist

        mean_latent = posterior.mode()
        mean_latent = mean_latent * 0.1
        # print(mean_latent.shape)
        z_quantized, fsq_dict = self.quantizer(mean_latent, n_steps=n_steps, inv_temperature=10)
        # print(mean_latent)
        # print(fsq_dict)
        # sys.exit()
        if not return_dict:
            # 返回一个元组 (tuple)
            return (z_quantized, fsq_dict)
            
        # ==================== ✅ 核心修复：返回一个普通字典 ====================
        return {
            "quantized_latent": z_quantized,
            "fsq_dict": fsq_dict
        }


    def decode(self, z: torch.Tensor, return_dict: bool = True, **kwargs) -> Union[torch.Tensor, Tuple]:

        outputs=super().decode(z=z, return_dict=True)
        
        if not return_dict:
            return (outputs.sample,)
            
        return outputs
            

    def forward(self, sample: torch.Tensor, n_steps: int = 0, return_dict: bool = True):
        """
        修改 forward 方法以处理来自 encode 的字典。
        """
        # 1. 编码和量化
        # `encode` 现在返回我们上面定义的字典
        encode_outputs = self.encode(sample, return_dict=True, n_steps=n_steps)
        
        # ==================== ✅ 核心修复：从字典中获取值 ====================
        z_quantized = encode_outputs["quantized_latent"]
        fsq_dict = encode_outputs["fsq_dict"]
        # ================================================================
        
        # 2. 解码
        reconstruction = self.decode(z_quantized, return_dict=True).sample
        reconstruction = self.mel_spectrogram_to_waveform(reconstruction)
        
        if not return_dict:
            return (reconstruction, fsq_dict)
            
        # forward 的返回保持不变，它本来就是一个字典，很完美
        return {
            "reconstruction": reconstruction,
            "fsq_dict": fsq_dict,
            "quantized_latent": z_quantized
        }

# class AutoencoderFSQ(nn.Module):
#     def __init__(
#         self,
#         ddconfig=None,
#         lossconfig=None,
#         batchsize=None,
#         embed_dim=None,
#         time_shuffle=1,
#         subband=1,
#         sampling_rate=16000,
#         ckpt_path=None,
#         reload_from_ckpt=None,
#         ignore_keys=[],
#         image_key="fbank",
#         colorize_nlabels=None,
#         monitor=None,
#         base_learning_rate=1e-5,
#         # FSQ specific parameters
#         levels=[8, 5, 5, 5],  # FSQ quantization levels
#         num_codebooks=1,
#         entropy_loss_weight=0.0,
#         commitment_loss_weight=0.25,
#         diversity_gamma=1.0,
#         use_projection=True,
#     ):
#         super().__init__()
#         self.automatic_optimization = False
#         assert (
#             "mel_bins" in ddconfig.keys()
#         ), "mel_bins is not specified in the Autoencoder config"
#         num_mel = ddconfig["mel_bins"]
#         self.image_key = image_key
#         self.sampling_rate = sampling_rate
#         self.encoder = Encoder(**ddconfig)
#         self.decoder = Decoder(**ddconfig)

#         self.loss = None
#         self.subband = int(subband)

#         if self.subband > 1:
#             print("Use subband decomposition %s" % self.subband)

#         # Modified for FSQ: no need for double_z anymore
#         self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
#         self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
#         # Initialize FSQ Regularizer
#         self.regularizer = FSQRegularizer(
#             levels=levels,
#             dim=embed_dim,
#             num_codebooks=num_codebooks,
#             entropy_loss_weight=entropy_loss_weight,
#             commitment_loss_weight=commitment_loss_weight,
#             diversity_gamma=diversity_gamma,
#             use_projection=use_projection,
#         )

#         if self.image_key == "fbank":
#             self.vocoder = get_vocoder(None, "cpu", num_mel)
#         self.embed_dim = embed_dim
#         if colorize_nlabels is not None:
#             assert type(colorize_nlabels) == int
#             self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
#         if monitor is not None:
#             self.monitor = monitor
#         if ckpt_path is not None:
#             self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
#         self.learning_rate = float(base_learning_rate)

#         self.time_shuffle = time_shuffle
#         self.reload_from_ckpt = reload_from_ckpt
#         self.reloaded = False
#         self.mean, self.std = None, None

#         self.feature_cache = None
#         self.flag_first_run = True
#         self.train_step = 0

#         self.logger_save_dir = None
#         self.logger_exp_name = None

#     def get_log_dir(self):
#         if self.logger_save_dir is None and self.logger_exp_name is None:
#             return os.path.join(self.logger.save_dir, self.logger._project)
#         else:
#             return os.path.join(self.logger_save_dir, self.logger_exp_name)

#     def set_log_dir(self, save_dir, exp_name):
#         self.logger_save_dir = save_dir
#         self.logger_exp_name = exp_name

#     def init_from_ckpt(self, path, ignore_keys=list()):
#         sd = torch.load(path, map_location="cpu")["state_dict"]
#         keys = list(sd.keys())
#         for k in keys:
#             for ik in ignore_keys:
#                 if k.startswith(ik):
#                     print("Deleting key {} from state_dict.".format(k))
#                     del sd[k]
#         self.load_state_dict(sd, strict=False)
#         print(f"Restored from {path}")

#     def encode(self, x, n_steps=0):
#         # Encode input to latent space
#         h = self.encoder(x)
#         # Project to FSQ embedding dimension
#         z = self.quant_conv(h)
        
#         # Reshape for FSQ regularizer: (B, C, H, W) -> (B, H, W, C) -> (B, H*W, C)
#         B, C, H, W = z.shape
#         z = z.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
#         # Apply FSQ quantization
#         z_quantized, fsq_dict = self.regularizer(z, n_steps=n_steps)

#         # Reshape back: (B, H*W, C) -> (B, H, W, C) -> (B, C, H, W)
#         z_quantized = z_quantized.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
#         return z_quantized, fsq_dict

#     def decode(self, z):
#         z = self.post_quant_conv(z)
#         dec = self.decoder(z)
#         return dec

#     def decode_to_waveform(self, dec):
#         from audioldm2.utilities.model import vocoder_infer

#         if self.image_key == "fbank":
#             dec = dec.squeeze(1).permute(0, 2, 1)
#             wav_reconstruction = vocoder_infer(dec, self.vocoder)
#         elif self.image_key == "stft":
#             dec = dec.squeeze(1).permute(0, 2, 1)
#             wav_reconstruction = self.wave_decoder(dec)
#         return wav_reconstruction

#     def visualize_latent(self, input):
#         import matplotlib.pyplot as plt

#         # for i in range(10):
#         #     zero_input = torch.zeros_like(input) - 11.59
#         #     zero_input[:,:,i * 16: i * 16 + 16,:16] += 13.59

#         #     posterior = self.encode(zero_input)
#         #     latent = posterior.sample()
#         #     avg_latent = torch.mean(latent, dim=1)[0]
#         #     plt.imshow(avg_latent.cpu().detach().numpy().T)
#         #     plt.savefig("%s.png" % i)
#         #     plt.close()

#         np.save("input.npy", input.cpu().detach().numpy())
#         # zero_input = torch.zeros_like(input) - 11.59
#         time_input = input.clone()
#         time_input[:, :, :, :32] *= 0
#         time_input[:, :, :, :32] -= 11.59

#         np.save("time_input.npy", time_input.cpu().detach().numpy())

#         posterior = self.encode(time_input)
#         latent = posterior.sample()
#         np.save("time_latent.npy", latent.cpu().detach().numpy())
#         avg_latent = torch.mean(latent, dim=1)
#         for i in range(avg_latent.size(0)):
#             plt.imshow(avg_latent[i].cpu().detach().numpy().T)
#             plt.savefig("freq_%s.png" % i)
#             plt.close()

#         freq_input = input.clone()
#         freq_input[:, :, :512, :] *= 0
#         freq_input[:, :, :512, :] -= 11.59

#         np.save("freq_input.npy", freq_input.cpu().detach().numpy())

#         posterior = self.encode(freq_input)
#         latent = posterior.sample()
#         np.save("freq_latent.npy", latent.cpu().detach().numpy())
#         avg_latent = torch.mean(latent, dim=1)
#         for i in range(avg_latent.size(0)):
#             plt.imshow(avg_latent[i].cpu().detach().numpy().T)
#             plt.savefig("time_%s.png" % i)
#             plt.close()

#     def get_input(self, batch):
#         fname, text, label_indices, waveform, stft, fbank = (
#             batch["fname"],
#             batch["text"],
#             batch["label_vector"],
#             batch["waveform"],
#             batch["stft"],
#             batch["log_mel_spec"],
#         )
#         # if(self.time_shuffle != 1):
#         #     if(fbank.size(1) % self.time_shuffle != 0):
#         #         pad_len = self.time_shuffle - (fbank.size(1) % self.time_shuffle)
#         #         fbank = torch.nn.functional.pad(fbank, (0,0,0,pad_len))

#         ret = {}

#         ret["fbank"], ret["stft"], ret["fname"], ret["waveform"] = (
#             fbank.unsqueeze(1),
#             stft.unsqueeze(1),
#             fname,
#             waveform.unsqueeze(1),
#         )

#         return ret

#     def save_wave(self, batch_wav, fname, save_dir):
#         os.makedirs(save_dir, exist_ok=True)

#         for wav, name in zip(batch_wav, fname):
#             name = os.path.basename(name)

#             sf.write(os.path.join(save_dir, name), wav, samplerate=self.sampling_rate)

#     def get_last_layer(self):
#         return self.decoder.conv_out.weight

#     @torch.no_grad()
#     def log_images(self, batch, train=True, only_inputs=False, waveform=None, **kwargs):
#         # ... (similar to original, but adapted for FSQ)
#         log = dict()
#         x = batch.to(self.device)
#         if not only_inputs:
#             z_quantized, fsq_dict = self.encode(x)
#             xrec = self.decode(z_quantized)
#             log["samples"] = xrec
#             log["reconstructions"] = xrec
#         log["inputs"] = x
#         wavs = self._log_img(log, train=train, index=0, waveform=waveform)
#         return wavs

#     def _log_img(self, log, train=True, index=0, waveform=None):
#         images_input = self.tensor2numpy(log["inputs"][index, 0]).T
#         images_reconstruct = self.tensor2numpy(log["reconstructions"][index, 0]).T
#         images_samples = self.tensor2numpy(log["samples"][index, 0]).T

#         if train:
#             name = "train"
#         else:
#             name = "val"

#         if self.logger is not None:
#             self.logger.log_image(
#                 "img_%s" % name,
#                 [images_input, images_reconstruct, images_samples],
#                 caption=["input", "reconstruct", "samples"],
#             )

#         inputs, reconstructions, samples = (
#             log["inputs"],
#             log["reconstructions"],
#             log["samples"],
#         )

#         if self.image_key == "fbank":
#             wav_original, wav_prediction = synth_one_sample(
#                 inputs[index],
#                 reconstructions[index],
#                 labels="validation",
#                 vocoder=self.vocoder,
#             )
#             wav_original, wav_samples = synth_one_sample(
#                 inputs[index], samples[index], labels="validation", vocoder=self.vocoder
#             )
#             wav_original, wav_samples, wav_prediction = (
#                 wav_original[0],
#                 wav_samples[0],
#                 wav_prediction[0],
#             )
#         elif self.image_key == "stft":
#             wav_prediction = (
#                 self.decode_to_waveform(reconstructions)[index, 0]
#                 .cpu()
#                 .detach()
#                 .numpy()
#             )
#             wav_samples = (
#                 self.decode_to_waveform(samples)[index, 0].cpu().detach().numpy()
#             )
#             wav_original = waveform[index, 0].cpu().detach().numpy()

#         if self.logger is not None:
#             self.logger.experiment.log(
#                 {
#                     "original_%s"
#                     % name: wandb.Audio(
#                         wav_original, caption="original", sample_rate=self.sampling_rate
#                     ),
#                     "reconstruct_%s"
#                     % name: wandb.Audio(
#                         wav_prediction,
#                         caption="reconstruct",
#                         sample_rate=self.sampling_rate,
#                     ),
#                     "samples_%s"
#                     % name: wandb.Audio(
#                         wav_samples, caption="samples", sample_rate=self.sampling_rate
#                     ),
#                 }
#             )

#         return wav_original, wav_prediction, wav_samples

#     def tensor2numpy(self, tensor):
#         return tensor.cpu().detach().numpy()

#     def to_rgb(self, x):
#         assert self.image_key == "segmentation"
#         if not hasattr(self, "colorize"):
#             self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
#         x = F.conv2d(x, weight=self.colorize)
#         x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
#         return x


# class IdentityFirstStage(torch.nn.Module):
#     def __init__(self, *args, vq_interface=False, **kwargs):
#         self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
#         super().__init__()

#     def encode(self, x, *args, **kwargs):
#         return x

#     def decode(self, x, *args, **kwargs):
#         return x

#     def quantize(self, x, *args, **kwargs):
#         if self.vq_interface:
#             return x, None, [None, None, None]
#         return x

#     def forward(self, x, *args, **kwargs):
#         return x
