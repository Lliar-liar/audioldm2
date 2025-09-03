import torch
import torchaudio
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import entropy
import warnings

# 安装依赖（如果需要）：
# pip install torchvision torchaudio

class FADCalculator:
    """计算Fréchet Audio Distance (FAD)"""
    
    def __init__(self, model_name='vggish', device='cuda'):
        """
        初始化FAD计算器
        model_name: 'vggish' 或 'pann'
        """
        self.device = device
        self.model_name = model_name
        
        if model_name == 'vggish':
            self.model = self._load_vggish()
        elif model_name == 'pann':
            self.model = self._load_pann()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _load_vggish(self):
        """加载VGGish模型 (Google的音频特征提取器)"""
        try:
            # 方法1: 使用torchaudio的预训练模型
            from torchaudio.models import vggish
            model = vggish()
            model.eval()
            model.to(self.device)
            print("✓ 加载了torchaudio VGGish")
            return model
        except:
            print("! VGGish不可用，使用替代方案...")
            return self._load_simple_extractor()
    
    def _load_pann(self):
        """加载PANNs (更现代的音频特征提取器)"""
        try:
            # 需要安装: pip install panns-inference
            from panns_inference import AudioTagging
            model = AudioTagging(checkpoint_path=None, device=self.device)
            print("✓ 加载了PANNs")
            return model
        except:
            print("! PANNs不可用，使用替代方案...")
            return self._load_simple_extractor()
    
    def _load_simple_extractor(self):
        """简单的特征提取器作为后备方案"""
        class SimpleAudioFeatureExtractor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 使用预训练的wav2vec2或其他可用模型
                try:
                    from transformers import Wav2Vec2Model, Wav2Vec2Processor
                    self.processor = Wav2Vec2Processor.from_pretrained(
                        "facebook/wav2vec2-base"
                    )
                    self.model = Wav2Vec2Model.from_pretrained(
                        "facebook/wav2vec2-base"
                    )
                    self.model.eval()
                    print("✓ 使用Wav2Vec2作为特征提取器")
                except:
                    # 最后的后备：简单的频谱特征
                    print("✓ 使用频谱特征提取器")
                    self.processor = None
                    self.model = None
            
            def forward(self, audio, sr=16000):
                if self.model is not None:
                    inputs = self.processor(
                        audio, 
                        sampling_rate=sr, 
                        return_tensors="pt"
                    )
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    # 使用最后隐藏状态的平均作为特征
                    features = outputs.last_hidden_state.mean(dim=1)
                else:
                    # 简单的频谱特征
                    features = self._extract_spectral_features(audio, sr)
                return features
            
            def _extract_spectral_features(self, audio, sr):
                """提取简单的频谱统计特征"""
                # 计算mel频谱图
                mel_spec = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sr,
                    n_fft=2048,
                    hop_length=512,
                    n_mels=128
                )(torch.tensor(audio).float())
                
                # 提取统计特征
                features = []
                features.append(mel_spec.mean(dim=-1))  # 时间平均
                features.append(mel_spec.std(dim=-1))   # 时间标准差
                features.append(mel_spec.max(dim=-1)[0]) # 最大值
                features.append(mel_spec.min(dim=-1)[0]) # 最小值
                
                # 拼接所有特征
                return torch.cat(features, dim=-1)
        
        return SimpleAudioFeatureExtractor().to(self.device)
    
    def extract_features(self, audio, sr=16000):
        """
        提取音频特征
        audio: numpy array 或 torch tensor
        sr: 采样率
        """
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio).float()
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # 添加batch维度
        
        audio = audio.to(self.device)
        
        with torch.no_grad():
            if self.model_name == 'vggish':
                # VGGish需要特定的预处理
                if hasattr(self.model, 'forward'):
                    features = self.model(audio, sr)
                else:
                    features = self.model(audio)
            elif self.model_name == 'pann':
                features = self.model.inference(audio.cpu().numpy())[1]
                features = torch.tensor(features).to(self.device)
            else:
                features = self.model(audio.cpu().numpy(), sr)
        
        return features.cpu().numpy()
    
    def calculate_statistics(self, features):
        """计算特征的均值和协方差"""
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
    
    def calculate_fad(self, features1, features2):
        """
        计算两组特征之间的Fréchet距离
        这是FAD的核心计算
        """
        mu1, sigma1 = self.calculate_statistics(features1)
        mu2, sigma2 = self.calculate_statistics(features2)
        
        # 计算Fréchet距离
        diff = mu1 - mu2
        
        # 处理协方差矩阵
        if len(sigma1.shape) == 0:  # 标量情况
            sigma1 = np.array([[sigma1]])
            sigma2 = np.array([[sigma2]])
        elif len(sigma1.shape) == 1:  # 一维情况
            sigma1 = np.diag(sigma1)
            sigma2 = np.diag(sigma2)
        
        # 计算协方差乘积的平方根
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        
        # 数值稳定性：确保实数
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                warnings.warn(f"复数协方差矩阵，虚部最大值: {m}")
            covmean = covmean.real
        
        # Fréchet距离公式
        tr_covmean = np.trace(covmean)
        fad = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        
        return float(fad)
    
    def compute_fad_from_audio(self, audio1, audio2, sr=16000):
        """
        直接从音频计算FAD
        注意：FAD通常需要多个样本，这里是简化版本
        """
        # 提取特征
        features1 = self.extract_features(audio1, sr)
        features2 = self.extract_features(audio2, sr)
        
        # 对于单个样本，我们使用滑动窗口创建"伪批次"
        features1_batch = self._create_pseudo_batch(features1)
        features2_batch = self._create_pseudo_batch(features2)
        
        # 计算FAD
        fad = self.calculate_fad(features1_batch, features2_batch)
        
        return fad
    
    def _create_pseudo_batch(self, features):
        """为单样本创建伪批次用于FAD计算"""
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # 添加噪声创建变体
        batch = []
        for i in range(10):  # 创建10个轻微变体
            noise = np.random.normal(0, 0.01, features.shape)
            batch.append(features + noise)
        
        return np.vstack(batch)


def compute_all_metrics_with_fad(original_audio, reconstructed_audio, sr=16000):
    """
    计算所有指标，包括FAD
    """
    print("\n=== 计算完整评估指标 ===\n")
    
    results = {}
    
    # 1. 时域指标（参考用）
    print("计算时域指标...")
    results['waveform_l1'] = np.mean(np.abs(original_audio - reconstructed_audio))
    results['waveform_l2'] = np.mean((original_audio - reconstructed_audio) ** 2)
    
    # 2. 频域指标（重要）
    print("计算频域指标...")
    
    # Multi-resolution STFT
    stft_losses = []
    for n_fft in [2048, 1024, 512, 256, 128]:
        hop_length = n_fft // 4
        
        # 计算STFT
        orig_stft = np.abs(librosa.stft(original_audio, n_fft=n_fft, hop_length=hop_length))
        rec_stft = np.abs(librosa.stft(reconstructed_audio, n_fft=n_fft, hop_length=hop_length))
        
        # Spectral convergence
        spec_conv = np.linalg.norm(orig_stft - rec_stft, 'fro') / (np.linalg.norm(orig_stft, 'fro') + 1e-8)
        
        # Log magnitude loss  
        log_orig = np.log(orig_stft + 1e-8)
        log_rec = np.log(rec_stft + 1e-8)
        log_loss = np.mean(np.abs(log_orig - log_rec))
        
        stft_losses.append(spec_conv + log_loss)
    
    results['multi_resolution_stft'] = np.mean(stft_losses)
    results['spectral_convergence'] = spec_conv  # 使用最高分辨率的
    results['log_magnitude_loss'] = log_loss
    
    # 3. Mel频谱图损失
    print("计算Mel频谱图损失...")
    mel_orig = librosa.feature.melspectrogram(y=original_audio, sr=sr, n_mels=80)
    mel_rec = librosa.feature.melspectrogram(y=reconstructed_audio, sr=sr, n_mels=80)
    results['mel_l1_loss'] = np.mean(np.abs(mel_orig - mel_rec))
    results['mel_l2_loss'] = np.mean((mel_orig - mel_rec) ** 2)
    
    # 4. FAD (最重要！)
    print("计算FAD...")
    try:
        fad_calculator = FADCalculator(model_name='vggish', device='cuda' if torch.cuda.is_available() else 'cpu')
        fad_score = fad_calculator.compute_fad_from_audio(original_audio, reconstructed_audio, sr)
        results['fad'] = fad_score
        
        # 参考值解释
        fad_quality = "优秀" if fad_score < 2.0 else "良好" if fad_score < 5.0 else "一般" if fad_score < 10.0 else "较差"
        results['fad_quality'] = fad_quality
    except Exception as e:
        print(f"FAD计算失败: {e}")
        results['fad'] = None
    
    # 5. SI-SDR (可选)
    print("计算SI-SDR...")
    try:
        from mir_eval.separation import bss_eval_sources
        sdr, sir, sar, _ = bss_eval_sources(original_audio.reshape(1, -1), reconstructed_audio.reshape(1, -1))
        results['si_sdr'] = sdr[0]
    except:
        # 简单实现
        alpha = np.dot(reconstructed_audio, original_audio) / (np.dot(original_audio, original_audio) + 1e-8)
        results['si_sdr'] = 20 * np.log10(np.linalg.norm(alpha * original_audio) / (np.linalg.norm(alpha * original_audio - reconstructed_audio) + 1e-8))
    
    return results


def print_evaluation_results(results):
    """打印评估结果的格式化输出"""
    print("\n" + "="*60)
    print(" "*20 + "VAE重建评估报告")
    print("="*60)
    
    print("\n📊 主要指标 (Primary Metrics):")
    print("-"*40)
    
    if results.get('fad') is not None:
        print(f"FAD Score:                    {results['fad']:.2f} [{results.get('fad_quality', '未知')}]")
        print(f"  ├─ AudioLDM2 参考值:        ~2.0")
        print(f"  ├─ Stable Audio 参考值:     ~2.8")
        print(f"  └─ 评价: <2=优秀, <5=良好, <10=一般")
    
    print(f"\nMulti-Resolution STFT Loss:  {results['multi_resolution_stft']:.4f}")
    print(f"  └─ 正常范围: 2.0-4.0")
    
    print(f"\nSpectral Convergence:         {results['spectral_convergence']:.4f}")
    print(f"  └─ 正常范围: 0.5-1.0")
    
    print(f"\nMel L1 Loss:                  {results['mel_l1_loss']:.4f}")
    print(f"  └─ AudioLDM2范围: 0.05-0.15")
    
    print("\n📈 次要指标 (Secondary Metrics):")
    print("-"*40)
    
    print(f"Log Magnitude Loss:           {results['log_magnitude_loss']:.4f}")
    print(f"SI-SDR:                       {results.get('si_sdr', 0):.2f} dB")
    print(f"Mel L2 Loss:                  {results['mel_l2_loss']:.6f}")
    
    print("\n⚠️ 仅供参考 (时域指标不适用于VAE):")
    print("-"*40)
    print(f"Waveform L1:                  {results['waveform_l1']:.4f}")
    print(f"Waveform L2:                  {results['waveform_l2']:.6f}")
    
    print("\n" + "="*60)
    print("结论: ", end="")
    
    # 自动判断质量
    if results.get('fad') is not None and results['fad'] < 5.0:
        if results['multi_resolution_stft'] < 4.0:
            print("✅ VAE重建质量良好，符合预期！")
        else:
            print("⚠️ FAD良好但STFT偏高，可能需要检查")
    elif results['multi_resolution_stft'] < 4.0:
        print("✅ STFT损失正常，VAE工作正常")
    else:
        print("⚠️ 指标偏高，可能需要优化")
    
    print("="*60 + "\n")


# 使用示例
if __name__ == "__main__":
    # 假设你已经有了音频数据
    import librosa
    
    # 加载音频
    original, sr = librosa.load("original.wav", sr=16000)
    reconstructed, sr = librosa.load("reconstructed.wav", sr=16000)
    
    # 计算所有指标
    results = compute_all_metrics_with_fad(original, reconstructed, sr)
    
    # 打印格式化结果
    print_evaluation_results(results)
    
    # 也可以保存为JSON
    import json
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
