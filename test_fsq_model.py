# in test_fsq_model.py

import torch
# 导入您上面定义的新类
from audioldm2.latent_encoder.fsqvae import AutoencoderFSQ 

# --- 1. 使用 from_pretrained 加载模型 ---
# 它会自动下载 config.json 和权重，并用这些参数初始化您的 AutoencoderFSQ 类
print("Loading FSQ-VAE model from Hugging Face Hub...")
model = AutoencoderFSQ.from_pretrained(
    "cvssp/audioldm2", 
    subfolder="vae", 
    # 您可以在这里覆盖 config.json 中的参数，或添加新参数
    fsq_levels=[8, 8, 8, 8, 8], 
    fsq_commitment_loss_weight=0.1,
    torch_dtype=torch.float32,
).to("cuda")

# --- 2. 手动设置 FSQ 参数 (因为 from_pretrained 会忽略未知参数) ---
# model.quantizer.commitment_loss_weight = 0.1 # 示例

print("Model loaded successfully!")
# print(model) # 打印模型结构

dummy_waveform = torch.randn(1, 17600).to("cuda", torch.float32)

# ==================== ✅ 关键修复：添加归一化 ====================
# 将波形归一化到 [-1, 1] 区间
# 这是一个标准的 peak normalization
dummy_waveform = dummy_waveform / torch.max(torch.abs(dummy_waveform))
dummy_waveform=dummy_waveform.squeeze(0)
print(dummy_waveform.shape)
# =================================================================

# a. 编码为 Token ID
with torch.no_grad():
    outputs = model.encode(dummy_waveform)
    indices = outputs["fsq_dict"]["indices"]
    print("Encoded Token Indices Shape:", indices.shape)

# b. 完整的前向传播 (用于训练)
outputs = model(dummy_waveform)
reconstructed_mel = outputs['reconstruction']
loss = outputs['fsq_dict']['aux_loss']
print("Reconstructed Mel Spectrogram Shape:", reconstructed_mel.shape)
print("FSQ Auxiliary Loss:", loss.item())