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

from audioldm2.latent_encoder.fsqvae import AutoencoderFSQ 
from audioldm2.utilities.data.audio_finetune_dataset import AudioWaveformDataset 
from audioldm2.modules.audio_loss import AudioReconstructionLoss

import sys


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
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 初始化模型
        self.model = AutoencoderFSQ.from_pretrained(
            model_name,
            subfolder="vae",
            fsq_levels=fsq_levels,
            fsq_commitment_loss_weight=fsq_commitment_loss_weight,
            torch_dtype=torch.float32,
        )
        
        # 初始化损失函数
        audio_loss_config = audio_loss_config or {}
        self.audio_loss = AudioReconstructionLoss(**audio_loss_config)
        
        # 用于记录的变量
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # 假设batch是音频波形张量或字典
        if isinstance(batch, dict):
            audio = batch['audio']
            # 可能还有其他元数据
        else:
            audio = batch

        # 前向传播
        outputs = self.model(audio)
        fsq_dict=outputs["fsq_dict"]
        # 解析输出
        if isinstance(outputs, dict):
            reconstructed = outputs.get('reconstruction', outputs.get('output'))
            fsq_loss = outputs.get('fsq_loss', 0.0)
            kl_loss = outputs.get('kl_loss', 0.0)
            codes = outputs.get('codes', None)
        else:
            reconstructed = outputs
            fsq_loss = 0.0
            kl_loss = 0.0
            codes = None
        
        # 计算重建损失
        recon_loss, audio_loss_dict = self.audio_loss(
            pred_waveform=reconstructed, 
            true_waveform=audio, 
            pred_latent=None, 
            true_latent= None, 
            global_step = self.global_step,
            optimizer_idx = 0
        )
        
        # 总损失
        
        total_loss = recon_loss + fsq_loss + kl_loss
        
        # 记录指标
        self.log('train/total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/recon_loss', recon_loss, on_step=True, on_epoch=True)
      
        self.log_dict(audio_loss_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log('train/aux_loss', fsq_dict.aux_loss, on_step=True, on_epoch=True)
  
        
        # 如果有codes，记录codebook使用率
        if codes is not None:
            unique_codes = torch.unique(codes).numel()
            total_codes = codes.numel()
            codebook_usage = unique_codes / total_codes
            self.log('train/codebook_usage', codebook_usage, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            audio = batch['audio']
        else:
            audio = batch
            
        # 前向传播
        print(audio.shape)
        with torch.no_grad():
            outputs = self.model(audio)
        
        fsq_dict=outputs["fsq_dict"]
        # 解析输出
        if isinstance(outputs, dict):
            reconstructed = outputs.get('reconstruction', outputs.get('output'))
            fsq_loss = outputs.get('fsq_loss', 0.0)
            kl_loss = outputs.get('kl_loss', 0.0)
            codes = outputs.get('codes', None)
        else:
            reconstructed = outputs
            fsq_loss = 0.0
            kl_loss = 0.0
            codes = None
        
        # 计算重建损失
        # print(reconstructed.shape)
        # print(audio.shape)
        recon_loss, audio_loss_dict = self.audio_loss(
            pred_waveform=reconstructed, 
            true_waveform=audio, 
            pred_latent=None, 
            true_latent= None, 
            global_step = self.global_step,
            optimizer_idx = 0
        )
        # print(recon_loss, fsq_loss, kl_loss)
        # 总损失
        total_loss = recon_loss + fsq_loss + kl_loss
        
        # 记录指标
        self.log('val/total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/recon_loss', recon_loss, on_step=False, on_epoch=True)

        self.log_dict(audio_loss_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log('train/aux_loss', fsq_dict.aux_loss, on_step=True, on_epoch=True)
        print(fsq_dict.aux_loss)
  
        
        # 记录codebook使用率
        if codes is not None:
            unique_codes = torch.unique(codes).numel()
            total_codes = codes.numel()
            codebook_usage = unique_codes / total_codes
            self.log('val/codebook_usage', codebook_usage, on_step=False, on_epoch=True)
        
        return total_loss
    
    def configure_optimizers(self):
        # 优化器
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
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
        elif self.hparams.lr_scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
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
    
    # 设置回调函数
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
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val/total_loss",
            patience=10,
            mode="min",
            verbose=True,
        ),
    ]
    
    # 设置日志记录器
    loggers = [
        TensorBoardLogger(
            save_dir=args.output_dir,
            name=args.experiment_name,
        )
    ]
    
    if args.use_wandb:
        loggers.append(
            WandbLogger(
                project="audio-vae-fsq",
                name=args.experiment_name,
                save_dir=args.output_dir,
            )
        )
    
    # 初始化训练器
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,  # 可以修改为多GPU训练
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        deterministic=False, 
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        
    )
    
    # 开始训练
    trainer.fit(
        model,
        data_module,
        ckpt_path=args.resume_from_checkpoint
    )
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, args.experiment_name, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()