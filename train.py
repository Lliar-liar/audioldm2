import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
import argparse
from typing import Optional, Dict, Any
import uuid

from audioldm2.latent_encoder.fsqvae import AutoencoderFSQ 
from audioldm2.utilities.data.audio_finetune_dataset import AudioWaveformDataset 
from audioldm2.modules.audio_loss import AudioReconstructionLoss
from pytorch_lightning.strategies import DDPStrategy
import sys
import time

class AudioVAEFSQLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "cvssp/audioldm2",
        fsq_levels: list = [8, 8, 8, 8, 8],
        fsq_commitment_loss_weight: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        lr_scheduler_type: str = "cosine",
        warmup_steps: int = 1000,
        audio_loss_config: Optional[Dict[str, Any]] = None,
        aux_loss_weight: float = 1.0,
        kl_loss_weight: float = 0.1,
        gradient_checkpointing: bool = False,
        freeze_vae_encoder: bool = False,
        freeze_vae_decoder: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.aux_loss_weight = aux_loss_weight
        self.kl_loss_weight = kl_loss_weight
        
        # 初始化模型 - 使用新的AutoencoderFSQ
        self.model = AutoencoderFSQ(
            model_name=model_name,
            subfolder="vae",  # 现在这个参数传给内部的from_pretrained
            fsq_levels=fsq_levels,
            fsq_commitment_loss_weight=fsq_commitment_loss_weight,
            torch_dtype=torch.float32,
        )
        
        # 可选：冻结VAE的某些部分
        if freeze_vae_encoder:
            for param in self.model.vae.encoder.parameters():
                param.requires_grad = False
            print("Froze VAE encoder parameters")
            
        if freeze_vae_decoder:
            for param in self.model.vae.decoder.parameters():
                param.requires_grad = False
            print("Froze VAE decoder parameters")
        
        # 可选：梯度检查点以节省内存
        if gradient_checkpointing:
            self.model.vae.enable_gradient_checkpointing()
        
        # 初始化损失函数
        audio_loss_config = audio_loss_config or {}
        self.audio_loss = AudioReconstructionLoss(**audio_loss_config)
        
        # 用于记录的变量
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def forward(self, x, n_steps=None):
        # 传递当前的训练步数给模型（用于FSQ温度调度）
        if n_steps is None:
            n_steps = self.global_step
        return self.model(x, n_steps=n_steps)
    
    def training_step(self, batch, batch_idx):
        # 解析输入
        if isinstance(batch, dict):
            audio = batch['audio']
            metadata = {k: v for k, v in batch.items() if k != 'audio'}
        else:
            audio = batch
            metadata = {}
        
        # 前向传播，传入当前步数用于FSQ温度调度
        outputs = self.model(audio, n_steps=self.global_step, return_dict=True)
        
        # 解析输出
        reconstructed = outputs['reconstruction']
        fsq_dict = outputs['fsq_dict']
        posterior = outputs.get('posterior', fsq_dict.get('posterior'))
        z_quantized = outputs['quantized_latent']
        z_unquantized = outputs.get('unquantized_latent', fsq_dict.get('unquantized'))
        
        # 处理批次维度（如果需要）
        if reconstructed.dim() > audio.dim():
            reconstructed = reconstructed.squeeze(0)
        
        # 1. 计算重建损失
        recon_loss, audio_loss_dict = self.audio_loss(
            pred_waveform=reconstructed, 
            true_waveform=audio, 
            pred_latent=z_quantized,  # 现在可以传入latent
            true_latent=z_unquantized,  # 用于额外的latent space损失
            global_step=self.global_step,
            optimizer_idx=0
        )
        
        # 2. KL散度损失（如果有posterior）
        kl_loss = torch.tensor(0.0).to(audio.device)
        if posterior is not None:
            try:
                kl_loss = posterior.kl().mean()
            except:
                kl_loss = torch.tensor(0.0).to(audio.device)
        
        # 3. FSQ辅助损失（commitment loss等）
        aux_loss = fsq_dict.get("aux_loss", torch.tensor(0.0).to(audio.device))
        
        # 确保所有损失都是标量
        if aux_loss.numel() > 1:
            aux_loss = aux_loss.mean()
        if kl_loss.numel() > 1:
            kl_loss = kl_loss.mean()
        
        # 4. 总损失
        total_loss = recon_loss + self.kl_loss_weight * kl_loss + self.aux_loss_weight * aux_loss
        
        # 5. 记录指标
        self.log('train/total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/recon_loss', recon_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/kl_loss', kl_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/aux_loss', aux_loss, on_step=True, on_epoch=True, sync_dist=True)
        
        # 记录音频损失的各个组成部分
        for key, value in audio_loss_dict.items():
            self.log(f'train/{key}', value, on_step=True, on_epoch=True, sync_dist=True)
        
        # 6. 计算并记录FSQ相关指标
        if 'codes' in fsq_dict:
            codes = fsq_dict['codes']
            unique_codes = torch.unique(codes).numel()
            total_codes = codes.numel()
            codebook_usage = unique_codes / total_codes
            self.log('train/codebook_usage', codebook_usage, on_step=True, on_epoch=True, sync_dist=True)
        
        # 记录量化相关指标
        if z_quantized is not None and z_unquantized is not None:
            quantization_error = F.mse_loss(z_quantized.detach(), z_unquantized.detach())
            self.log('train/quantization_error', quantization_error, on_step=True, on_epoch=True, sync_dist=True)
        
        # 记录梯度范数（用于调试）
        if self.global_step % 100 == 0:
            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            self.log('train/grad_norm', grad_norm, on_step=True, sync_dist=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # 解析输入
        if isinstance(batch, dict):
            audio = batch['audio']
        else:
            audio = batch
        
        # 只在第一个batch打印shape信息
        if batch_idx == 0 and self.trainer.global_rank == 0:
            print(f"Validation audio shape: {audio.shape}")
        
        # 前向传播（评估模式）
        with torch.no_grad():
            outputs = self.model(audio, n_steps=self.global_step, return_dict=True)
        
        # 解析输出
        reconstructed = outputs['reconstruction']
        fsq_dict = outputs['fsq_dict']
        posterior = outputs.get('posterior', fsq_dict.get('posterior'))
        z_quantized = outputs['quantized_latent']
        z_unquantized = outputs.get('unquantized_latent', fsq_dict.get('unquantized'))
        
        # 处理批次维度
        if reconstructed.dim() > audio.dim():
            reconstructed = reconstructed.squeeze(0)
        
        # 1. 计算重建损失
        recon_loss, audio_loss_dict = self.audio_loss(
            pred_waveform=reconstructed, 
            true_waveform=audio, 
            pred_latent=z_quantized,
            true_latent=z_unquantized,
            global_step=self.global_step,
            optimizer_idx=0
        )
        
        # 2. KL损失
        kl_loss = torch.tensor(0.0).to(audio.device)
        if posterior is not None:
            try:
                kl_loss = posterior.kl().mean()
            except:
                kl_loss = torch.tensor(0.0).to(audio.device)
        
        # 3. FSQ损失
        aux_loss = fsq_dict.get("aux_loss", torch.tensor(0.0).to(audio.device))
        
        # 确保所有损失都是标量
        if aux_loss.numel() > 1:
            aux_loss = aux_loss.mean()
        if kl_loss.numel() > 1:
            kl_loss = kl_loss.mean()
        
        # 4. 总损失
        total_loss = recon_loss + self.kl_loss_weight * kl_loss + self.aux_loss_weight * aux_loss
        
        # 5. 记录指标
        self.log('val/total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/recon_loss', recon_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/kl_loss', kl_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/aux_loss', aux_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        # 记录音频损失组成
        for key, value in audio_loss_dict.items():
            self.log(f'val/{key}', value, on_step=False, on_epoch=True, sync_dist=True)
        
        # 6. FSQ指标
        if 'codes' in fsq_dict:
            codes = fsq_dict['codes']
            unique_codes = torch.unique(codes).numel()
            total_codes = codes.numel()
            codebook_usage = unique_codes / total_codes
            self.log('val/codebook_usage', codebook_usage, on_step=False, on_epoch=True, sync_dist=True)
        
        # 量化误差
        if z_quantized is not None and z_unquantized is not None:
            quantization_error = F.mse_loss(z_quantized, z_unquantized)
            self.log('val/quantization_error', quantization_error, on_step=False, on_epoch=True, sync_dist=True)
        
        # 每个epoch保存一些音频样本用于可视化
        if batch_idx == 0 and self.logger:
            # 选择前几个样本
            num_samples = min(3, audio.shape[0])
            for i in range(num_samples):
                # 记录原始和重建的音频
                if hasattr(self.logger, 'experiment'):
                    self.logger.experiment.add_audio(
                        f'audio/original_{i}',
                        audio[i].cpu().numpy(),
                        self.global_step,
                        sample_rate=self.model.sampling_rate
                    )
                    self.logger.experiment.add_audio(
                        f'audio/reconstructed_{i}',
                        reconstructed[i].cpu().numpy(),
                        self.global_step,
                        sample_rate=self.model.sampling_rate
                    )
        
        return total_loss
    
    def configure_optimizers(self):
        # 分离参数组（可选）
        vae_params = list(self.model.vae.parameters())
        fsq_params = list(self.model.quantizer.parameters())
        
        # 可以为不同的参数组设置不同的学习率
        param_groups = [
            {'params': vae_params, 'lr': self.hparams.learning_rate},
            {'params': fsq_params, 'lr': self.hparams.learning_rate * 2}  # FSQ可能需要更高的学习率
        ]
        
        # 优化器
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        if self.hparams.lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer else 100,
                eta_min=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                }
            }
        elif self.hparams.lr_scheduler_type == "cosine_with_warmup":
            # 带warmup的cosine调度
            from torch.optim.lr_scheduler import LambdaLR
            
            def lr_lambda(step):
                if step < self.hparams.warmup_steps:
                    return step / self.hparams.warmup_steps
                else:
                    progress = (step - self.hparams.warmup_steps) / (self.trainer.estimated_stepping_batches - self.hparams.warmup_steps)
                    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
            
            scheduler = LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                }
            }
        elif self.hparams.lr_scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/total_loss",
                    "interval": "epoch",
                }
            }
        else:
            return optimizer
    
    def on_train_epoch_end(self):
        """在每个epoch结束时记录额外的信息"""
        # 可以在这里添加额外的日志或检查点逻辑
        pass
    
    def on_validation_epoch_end(self):
        """在验证epoch结束时记录额外的信息"""
        pass



class AudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv_path: str,
        val_csv_path: str,
        source_dir: str,  # 添加source_dir参数
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        audio_waveform_params: Optional[Dict[str, Any]] = None,  # 添加音频波形参数
        **dataset_kwargs
    ):
        super().__init__()
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.source_dir = source_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # 默认音频波形参数
        self.audio_waveform_params = audio_waveform_params or {
            'sample_rate': 16000,
            'duration': 10.0,  # 秒
            'n_mels': 64,
            'hop_length': 160,
            'n_fft': 1024,
            # 添加其他需要的参数
        }
        self.dataset_kwargs = dataset_kwargs
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = AudioWaveformDataset(
                meta_path=self.train_csv_path,
                source_dir=self.source_dir,
                audio_waveform_params=self.audio_waveform_params,
                **self.dataset_kwargs
            )
            self.val_dataset = AudioWaveformDataset(
                meta_path=self.val_csv_path,
                source_dir=self.source_dir,
                audio_waveform_params=self.audio_waveform_params,
                **self.dataset_kwargs
            )
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=False
        )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Audio VAE with FSQ")
    
    # 数据相关参数
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to validation CSV file")
    parser.add_argument("--source_dir", type=str, required=True, help="Source directory for audio files")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # 音频参数
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--duration", type=float, default=10.0, help="Audio duration in seconds")
    parser.add_argument("--n_mels", type=int, default=64, help="Number of mel bins")
    parser.add_argument("--hop_length", type=int, default=160, help="Hop length for STFT")
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT size")
    
    # 模型相关参数
    parser.add_argument("--model_name", type=str, default="cvssp/audioldm2", help="Pretrained model name")
    parser.add_argument("--fsq_levels", type=int, nargs="+", default=[8, 8, 8, 8, 8], help="FSQ levels")
    parser.add_argument("--fsq_commitment_loss_weight", type=float, default=0.1, help="FSQ commitment loss weight")
    
    # 训练相关参数
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps")
    
    # 日志和保存相关参数
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="audio_vae_fsq", help="Experiment name")
    parser.add_argument("--save_top_k", type=int, default=3, help="Save top k checkpoints")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="Log every n steps")
    parser.add_argument("--val_check_interval", type=float, default=1.0, help="Validation check interval")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--precision", type=str, default="32", choices=["16", "32", "bf16"], help="Training precision")
    parser.add_argument("--target_length", type=int, default=17600)
    
    args = parser.parse_args()
    
    # 设置随机种子
    pl.seed_everything(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取当前进程的rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    
    # 准备音频波形参数
    audio_waveform_params = {
        'sample_rate': args.sample_rate,
        'duration': args.duration,
        'n_mels': args.n_mels,
        'hop_length': args.hop_length,
        'n_fft': args.n_fft,
        "target_length": args.target_length,
    }
    
    # 初始化数据模块
    data_module = AudioDataModule(
        train_csv_path=args.train_csv,
        val_csv_path=args.val_csv,
        source_dir=args.source_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        audio_waveform_params=audio_waveform_params,
    )
    
    # 初始化模型
    model = AudioVAEFSQLightningModule(
        model_name=args.model_name,
        fsq_levels=args.fsq_levels,
        fsq_commitment_loss_weight=args.fsq_commitment_loss_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # 设置回调函数 - 基础回调
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, args.experiment_name, "checkpoints"),
            filename="epoch_{epoch:02d}_val_loss_{val/total_loss:.4f}",
            monitor="val/total_loss",
            mode="min",
            save_top_k=args.save_top_k,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/total_loss",
            patience=10,
            mode="min",
            verbose=True,
        ),
    ]
    
    # 设置日志记录器 - 只在主进程创建
    loggers = []
    
    if args.use_wandb and global_rank == 0:  # 只在主进程创建WandB logger
        # 为这次运行创建唯一的ID，避免resume冲突
        unique_id = f"{args.experiment_name}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        unique_run_name = f"{args.experiment_name}-{int(time.time())}"
        
        loggers.append(
            WandbLogger(
                project="audio-vae-fsq",
                name=unique_run_name,
                id=unique_id,  # 使用唯一ID避免冲突
                save_dir=args.output_dir,
                resume="never",  # 不resume
                reinit=True  # 重新初始化
            )
        )
        
        # 只在有logger的时候添加LearningRateMonitor
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    
    # 初始化训练器
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=loggers or None,  # 使用None而不是False
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        deterministic=False, 
        enable_checkpointing=True,
        enable_progress_bar=True if global_rank == 0 else False,  # 只在主进程显示进度条
        enable_model_summary=True if global_rank == 0 else False,  # 只在主进程显示模型摘要
    )
    
    # 开始训练
    trainer.fit(
        model,
        data_module,
        ckpt_path=args.resume_from_checkpoint
    )
    
    # 保存最终模型 - 只在主进程保存
    if global_rank == 0:
        final_model_path = os.path.join(args.output_dir, args.experiment_name, "final_model.pt")
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
