import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from typing import Dict, Union, List, Optional
from audioldm2.modules.discriminator import AudioDiscriminator1D, weights_init
import sys

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


class AudioDiscriminator1DWithFeatures(AudioDiscriminator1D):
    """扩展判别器以支持特征提取"""
    
    def forward_with_features(self, x):
        """前向传播并返回中间特征"""
        features = []
        
        # 获取所有层
        layers = list(self.children())
        
        # 如果判别器使用Sequential包装
        if len(layers) == 1 and isinstance(layers[0], nn.Sequential):
            layers = list(layers[0].children())
        
        # 逐层前向传播
        for i, layer in enumerate(layers):
            x = layer(x)
            # 保存卷积层后的特征（跳过激活函数等）
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                features.append(x)
        
        return x, features
    
    def get_features(self, x):
        """仅获取特征（用于特征匹配）"""
        _, features = self.forward_with_features(x)
        return features


class AudioReconstructionLoss(nn.Module):
    """优化后的音频重建损失"""
    def __init__(self,
                # 重建损失权重
                latent_loss_weight: float = 1.0,
                waveform_l1_loss_weight: float = 1.0,
                multi_res_spec_loss_weight: float = 45.0,
                mel_perceptual_weight: float = 10.0,
                
                # GAN相关权重
                gan_loss_weight: float = 0.1,
                gan_start_step: int = 10000,
                gan_feature_loss_weight: float = 2.0,
                
                # 模型参数
                latent_len: int = 28,
                audio_len: int = 17600,
                
                # 损失函数配置
                stft_params: Optional[Dict] = None,
                mel_params: Optional[Dict] = None,
                discriminator_params: Optional[Dict] = None,
                
                # 训练策略
                use_progressive_weights: bool = False,
                use_feature_matching: bool = True,  # 是否使用特征匹配
                ):
        super().__init__()
        
        # 保存权重
        self.latent_loss_weight = latent_loss_weight
        self.waveform_l1_loss_weight = waveform_l1_loss_weight
        self.multi_res_spec_loss_weight = multi_res_spec_loss_weight
        self.mel_perceptual_weight = mel_perceptual_weight
        self.gan_loss_weight = gan_loss_weight
        self.gan_start_step = gan_start_step
        self.gan_feature_loss_weight = gan_feature_loss_weight
        
        # 模型参数
        self.audio_len = audio_len
        self.latent_len = latent_len
        self.use_progressive_weights = use_progressive_weights
        self.use_feature_matching = use_feature_matching
        
        # 初始化损失函数
        # 1. Latent损失
        self.latent_loss_fn = nn.L1Loss() if latent_loss_weight > 0 else None
        
        # 2. 波形L1损失
        self.waveform_l1_loss_fn = nn.L1Loss() if waveform_l1_loss_weight > 0 else None
        
        # 3. 多分辨率STFT损失
        if multi_res_spec_loss_weight > 0:
            stft_params = stft_params or {
                'fft_sizes': [512, 1024, 2048],
                'hop_sizes': [160, 320, 640],
                'win_lengths': [512, 1024, 2048],
                'mag_weight': 1.0,
                'log_mag_weight': 1.0
            }
            self.multi_res_spec_loss_fn = MultiResolutionSTFTLoss(**stft_params)
        else:
            self.multi_res_spec_loss_fn = None
            
        # 4. 梅尔谱感知损失
        if mel_perceptual_weight > 0:
            mel_params = mel_params or {
                'sample_rate': 16000,
                'n_fft': 1024,
                'hop_length': 160,
                'n_mels': 64,
                'fmin': 0.0,
                'fmax': 8000.0
            }
            self.mel_loss_fn = MelSpectrogramLoss(**mel_params)
        else:
            self.mel_loss_fn = None
            
        # 5. 判别器
        discriminator_params = discriminator_params or {
            'input_nc': 1,
            'ndf': 32,
            'n_layers': 4
        }
        
        # 根据是否需要特征匹配选择判别器类型
        if self.use_feature_matching:
            self.discriminator = AudioDiscriminator1DWithFeatures(**discriminator_params)
        else:
            self.discriminator = AudioDiscriminator1D(**discriminator_params)
        
        self.discriminator.apply(weights_init)

    def get_trainable_parameters(self):
        """返回判别器的可训练参数"""
        return list(self.discriminator.parameters())
    
    def get_current_weights(self, global_step: int) -> Dict[str, float]:
        """渐进式调整权重（可选）"""
        if not self.use_progressive_weights:
            return {
                'latent': self.latent_loss_weight,
                'waveform': self.waveform_l1_loss_weight,
                'spec': self.multi_res_spec_loss_weight,
                'mel': self.mel_perceptual_weight,
                'gan': self.gan_loss_weight,
                'feature': self.gan_feature_loss_weight
            }
        
        # 渐进式权重策略
        if global_step < 5000:
            # 阶段1: 专注基础重建
            return {
                'latent': 1.0,
                'waveform': 0.5,
                'spec': 50.0,
                'mel': 10.0,
                'gan': 0.0,
                'feature': 0.0
            }
        elif global_step < 10000:
            # 阶段2: 平衡各项损失
            return {
                'latent': 1.0,
                'waveform': 1.0,
                'spec': 45.0,
                'mel': 10.0,
                'gan': 0.0,
                'feature': 0.0
            }
        else:
            # 阶段3: 加入GAN微调
            return {
                'latent': 1.0,
                'waveform': 1.0,
                'spec': 45.0,
                'mel': 10.0,
                'gan': 0.1,
                'feature': 2.0
            }

    def _hinge_d_loss(self, logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
        """Hinge判别器损失"""
        loss_real = torch.mean(torch.relu(1. - logits_real))
        loss_fake = torch.mean(torch.relu(1. + logits_fake))
        return 0.5 * (loss_real + loss_fake)

    def _hinge_g_loss(self, logits_fake: torch.Tensor) -> torch.Tensor:
        """Hinge生成器损失"""
        return -torch.mean(logits_fake)
    
    def _feature_matching_loss(self, pred_waveform: torch.Tensor, true_waveform: torch.Tensor) -> torch.Tensor:
        """特征匹配损失（安全版本）"""
        if not self.use_feature_matching:
            return torch.tensor(0.0, device=pred_waveform.device)
        
        try:
            # 使用forward_with_features获取特征
            _, feat_pred = self.discriminator.forward_with_features(pred_waveform)
            _, feat_true = self.discriminator.forward_with_features(true_waveform.detach())
            
            if len(feat_pred) == 0 or len(feat_true) == 0:
                return torch.tensor(0.0, device=pred_waveform.device)
            
            loss = 0
            for fp, ft in zip(feat_pred, feat_true):
                loss += F.l1_loss(fp, ft)
            
            return loss / len(feat_pred)
            
        except Exception as e:
            print(f"Warning: Feature matching failed: {e}")
            return torch.tensor(0.0, device=pred_waveform.device)

    def forward(self,
                pred_latent: torch.Tensor,
                true_latent: torch.Tensor,
                pred_waveform: torch.Tensor,
                true_waveform: torch.Tensor,
                global_step: int,
                optimizer_idx: int,  # 0 for generator, 1 for discriminator
               ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        # 确保波形有正确的维度
        if pred_waveform.dim() == 1:
            pred_waveform = pred_waveform.unsqueeze(0).unsqueeze(0)
        elif pred_waveform.dim() == 2:
            pred_waveform = pred_waveform.unsqueeze(0)
            
        log_dict = {}
        
        # 获取当前权重
        weights = self.get_current_weights(global_step)
        
        # ========== 生成器损失 ==========
        if optimizer_idx == 0:
            total_loss = 0.0
            
            # 1. Latent损失
            if self.latent_loss_fn is not None and pred_latent is not None:
                min_latent_len = min(self.latent_len, pred_latent.shape[-2], true_latent.shape[-2])
                pred_latent_crop = pred_latent[..., :min_latent_len, :]
                true_latent_crop = true_latent[..., :min_latent_len, :]
                
                latent_loss = self.latent_loss_fn(pred_latent_crop, true_latent_crop)
                weighted_loss = weights['latent'] * latent_loss
                total_loss += weighted_loss
                
                log_dict['audio/latent_loss'] = latent_loss.detach()
                log_dict['audio/weighted_latent_loss'] = weighted_loss.detach()
            
            # 对齐波形长度
            min_len = min(pred_waveform.shape[-1], true_waveform.shape[-1], self.audio_len)
            pred_waveform = pred_waveform[..., :min_len]
            true_waveform = true_waveform[..., :min_len]
            
            # 2. 波形L1损失
            if self.waveform_l1_loss_fn is not None:
                waveform_l1_loss = self.waveform_l1_loss_fn(pred_waveform, true_waveform)
                weighted_loss = weights['waveform'] * waveform_l1_loss
                total_loss += weighted_loss
                
                log_dict['audio/waveform_l1_loss'] = waveform_l1_loss.detach()
                log_dict['audio/weighted_waveform_l1_loss'] = weighted_loss.detach()
            
            # 3. 多分辨率STFT损失
            if self.multi_res_spec_loss_fn is not None:
                multi_res_spec_loss = self.multi_res_spec_loss_fn(pred_waveform, true_waveform)
                weighted_loss = weights['spec'] * multi_res_spec_loss
                total_loss += weighted_loss
                
                log_dict['audio/multi_res_spec_loss'] = multi_res_spec_loss.detach()
                log_dict['audio/weighted_multi_res_spec_loss'] = weighted_loss.detach()
            
            # 4. 梅尔谱感知损失
            if self.mel_loss_fn is not None:
                mel_loss = self.mel_loss_fn(pred_waveform, true_waveform)
                weighted_loss = weights['mel'] * mel_loss
                total_loss += weighted_loss
                
                log_dict['audio/mel_perceptual_loss'] = mel_loss.detach()
                log_dict['audio/weighted_mel_loss'] = weighted_loss.detach()
            
            # 5. GAN相关损失
            if weights['gan'] > 0 and global_step >= self.gan_start_step:
                # 对抗损失
                logits_fake = self.discriminator(pred_waveform)
                g_loss_gan = self._hinge_g_loss(logits_fake)
                weighted_loss = weights['gan'] * g_loss_gan
                total_loss += weighted_loss
                
                log_dict['audio/g_loss_gan'] = g_loss_gan.detach()
                log_dict['audio/weighted_g_loss_gan'] = weighted_loss.detach()
                
                # 特征匹配损失（如果启用并且权重>0）
                if self.use_feature_matching and weights['feature'] > 0:
                    feature_loss = self._feature_matching_loss(pred_waveform, true_waveform)
                    weighted_loss = weights['feature'] * feature_loss
                    total_loss += weighted_loss
                    
                    log_dict['audio/feature_matching_loss'] = feature_loss.detach()
                    log_dict['audio/weighted_feature_loss'] = weighted_loss.detach()
            
            log_dict['audio/total_loss'] = total_loss.detach()
            
            return total_loss, log_dict
        
        # ========== 判别器损失 ==========
        elif optimizer_idx == 1:
            if weights['gan'] > 0 and global_step >= self.gan_start_step:
                # 对齐长度
                min_len = min(pred_waveform.shape[-1], true_waveform.shape[-1], self.audio_len)
                pred_w = pred_waveform[..., :min_len]
                true_w = true_waveform[..., :min_len]
                
                # 确保维度正确
                if true_w.dim() == 2:
                    true_w = true_w.unsqueeze(1)
                if pred_w.dim() == 2:
                    pred_w = pred_w.unsqueeze(1)
                
                # 计算判别器损失
                logits_real = self.discriminator(true_w.detach())
                logits_fake = self.discriminator(pred_w.detach())
                d_loss = self._hinge_d_loss(logits_real, logits_fake)
                
                log_dict['audio/d_loss'] = d_loss.detach()
                log_dict['audio/logits_real'] = logits_real.detach().mean()
                log_dict['audio/logits_fake'] = logits_fake.detach().mean()
                
                return d_loss, log_dict
            else:
                # GAN还未启动
                return torch.tensor(0.0, device=pred_waveform.device, requires_grad=True), log_dict
