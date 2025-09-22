from abc import abstractmethod
from functools import cache
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, reduce, unpack
from torch import Tensor, int32
from torch.cuda.amp import autocast

from audioldm2.modules.distributions import DiagonalGaussianDistribution


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


def log(t, eps=1e-5):
    return t.clamp(min=eps).log()


def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)


def maybe_distributed_mean(t):
    if not is_distributed():
        return t
    dist.all_reduce(t)
    t = t / dist.get_world_size()
    return t


@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1


class AbstractRegularizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError()

    @abstractmethod
    def get_trainable_parameters(self) -> Any:
        raise NotImplementedError()


class DiagonalGaussianRegularizer(AbstractRegularizer):
    def __init__(self, sample: bool = True):
        super().__init__()
        self.sample = sample

    def get_trainable_parameters(self) -> Any:
        yield from ()

    def forward(self, z: torch.Tensor, n_steps=None) -> Tuple[torch.Tensor, dict]:
        log = dict()
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log["kl_loss"] = kl_loss
        return z, log


class FSQRegularizer(AbstractRegularizer):
    # https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/finite_scalar_quantization.py
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        entropy_loss_weight: float = 0.0,
        entropy_loss_annealing_steps: int = 0,
        entropy_loss_annealing_factor: float = 1.0,
        commitment_loss_weight: float = 0.0,
        diversity_gamma: float = 1.0,
        use_projection: bool = True,

        
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)
        self.use_projection = use_projection
        self.scale = scale
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_annealing_steps = entropy_loss_annealing_steps
        self.entropy_loss_annealing_factor = entropy_loss_annealing_factor
        self.commitment_loss_weight = commitment_loss_weight
        self.diversity_gamma = diversity_gamma

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.use_projection and (self.dim != effective_codebook_dim)
        if has_projections:
            self.project_in = nn.Linear(self.dim, effective_codebook_dim).to(torch.float32)
            self.project_out = nn.Linear(effective_codebook_dim, self.dim).to(torch.float32)
        else:
            # 如果不使用投影，就用 Identity 占位
            # 并且我们断言输入维度必须等于有效的码本维度
            assert self.dim == effective_codebook_dim, (
                f"When use_projection is False, the input dimension 'dim' ({self.dim}) "
                f"must be equal to the effective codebook dimension (len(levels) * num_codebooks = {effective_codebook_dim})."
            )
            self.project_in = nn.Identity().to(torch.float32)
            self.project_out = nn.Identity().to(torch.float32)
            
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        self.register_buffer("codebook_usage", torch.zeros(self.num_codebooks, self.codebook_size, dtype=torch.long))

    def get_trainable_parameters(self) -> Any:
        return self.parameters()
    
    @torch.no_grad()
    def update_codebook_usage(self, indices):
        """Update the codebook usage statistics."""
        if self.num_codebooks == 1:
            indices = rearrange(indices, '... -> 1 ...')

        for i in range(self.num_codebooks):
            # Flatten the indices for the current codebook
            flat_indices = indices[..., i].flatten()
            # Use bincount to get the counts of each index
            counts = torch.bincount(flat_indices, minlength=self.codebook_size)
            # Update the codebook usage buffer
            self.codebook_usage[i] += counts.to(self.codebook_usage.device)

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(self, indices: Tensor, project_out=True) -> Tensor:
        """Inverse of `codes_to_indices`."""

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def calculate_entropy_loss_weight(self, n_steps):
        if n_steps >= self.entropy_loss_annealing_steps:
            return self.entropy_loss_weight
        start = self.entropy_loss_annealing_factor * self.entropy_loss_weight
        return start - (n_steps / self.entropy_loss_annealing_steps) * (start - self.entropy_loss_weight)

    @autocast(enabled=False)
    def forward(self, z: Tensor, inv_temperature: float = 100.0, n_steps: int = 0) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """
        # 检查输入
        print(f"[1] Input z: shape={z.shape}, finite={torch.isfinite(z).all()}, range=[{z.min():.4f}, {z.max():.4f}]")
        
        is_img_or_video = z.ndim >= 4
        if is_img_or_video:
            z = rearrange(z, "b d ... -> b ... d")
            z, ps = pack_one(z, "b * d")

        orig_dtype = z.dtype
        z = z.float()

        # 投影前
        print(f"[2] Before project_in: finite={torch.isfinite(z).all()}, range=[{z.min():.4f}, {z.max():.4f}]")
        z = self.project_in(z)
        # 投影后
        print(f"[3] After project_in: finite={torch.isfinite(z).all()}, range=[{z.min():.4f}, {z.max():.4f}]")

        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        with torch.autocast("cuda", enabled=False):
            
            original_input = z
            print(f"[4] original_input: finite={torch.isfinite(original_input).all()}, range=[{original_input.min():.4f}, {original_input.max():.4f}]")
            
            codes = self.quantize(z)
            print(f"[5] After quantize: finite={torch.isfinite(codes).all()}, range=[{codes.min():.4f}, {codes.max():.4f}]")
            
            indices = self.codes_to_indices(codes)
            print(f"[6] indices: finite={torch.isfinite(indices.float()).all()}, shape={indices.shape}")

            if self.training:
                self.update_codebook_usage(indices)

            if self.entropy_loss_weight > 0 or self.commitment_loss_weight > 0:
                # 检查implicit_codebook
                print(f"[7] implicit_codebook: finite={torch.isfinite(self.implicit_codebook).all()}, range=[{self.implicit_codebook.min():.4f}, {self.implicit_codebook.max():.4f}]")
                
                if self.implicit_codebook.dtype != original_input.dtype:
                    print(f"!!!!!!Converting implicit_codebook dtype from {self.implicit_codebook.dtype} to {original_input.dtype}")
                    self.implicit_codebook = self.implicit_codebook.to(original_input.dtype)
                
                # 计算距离
                distance = -2 * torch.einsum("... i d, j d -> ... i j", original_input, self.implicit_codebook)
                print(f"[8] distance: finite={torch.isfinite(distance).all()}, range=[{distance.min():.4f}, {distance.max():.4f}]")
                
                # 计算logits（softmax之前）
                logits = -distance * inv_temperature
                print(f"[9] logits (before softmax): finite={torch.isfinite(logits).all()}, range=[{logits.min():.4f}, {logits.max():.4f}], inv_temp={inv_temperature}")
                
                # Softmax
                prob = logits.softmax(dim=-1)
                print(f"[10] prob (after softmax): finite={torch.isfinite(prob).all()}, range=[{prob.min():.4f}, {prob.max():.4f}]")
                
                per_sample_probs = rearrange(prob, "b n ... -> (b n) ...")
                
                # 计算熵
                per_sample_entropy = entropy(per_sample_probs).mean()
                print(f"[11] per_sample_entropy: finite={torch.isfinite(per_sample_entropy).all()}, value={per_sample_entropy.item():.4f}")
                
                # distribution over all available tokens in the batch
                avg_prob = reduce(per_sample_probs, "... c d -> c d", "mean")
                avg_prob = maybe_distributed_mean(avg_prob)
                print(f"[12] avg_prob: finite={torch.isfinite(avg_prob).all()}, range=[{avg_prob.min():.4f}, {avg_prob.max():.4f}]")
                
                codebook_entropy = entropy(avg_prob).mean()
                print(f"[13] codebook_entropy: finite={torch.isfinite(codebook_entropy).all()}, value={codebook_entropy.item():.4f}")
                
                entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
                print(f"[14] entropy_aux_loss: finite={torch.isfinite(entropy_aux_loss).all()}, value={entropy_aux_loss.item():.4f}")
                
                # commit loss
                commit_loss = F.mse_loss(original_input, codes.detach(), reduction="none")
                print(f"[15] commit_loss (before mean): finite={torch.isfinite(commit_loss).all()}, range=[{commit_loss.min():.4f}, {commit_loss.max():.4f}]")
                
                commit_loss = commit_loss.mean()
                print(f"[16] commit_loss (after mean): finite={torch.isfinite(commit_loss).all()}, value={commit_loss.item():.4f}")
            else:
                entropy_aux_loss = per_sample_entropy = codebook_entropy = commit_loss = self.zero

        codes = rearrange(codes, "b n c d -> b n (c d)")
        out = self.project_out(codes)
        print(f"[17] After project_out: finite={torch.isfinite(out).all()}, range=[{out.min():.4f}, {out.max():.4f}]")

        # reconstitute image or video dimensions
        if is_img_or_video:
            out = unpack_one(out, ps, "b * d")
            out = rearrange(out, "b ... d -> b d ...")
            indices = unpack_one(indices, ps, "b * c")

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        aux_loss = (
            entropy_aux_loss * self.calculate_entropy_loss_weight(n_steps) + commit_loss * self.commitment_loss_weight
        )
        print(f"[18] Final aux_loss: finite={torch.isfinite(aux_loss).all()}, value={aux_loss.item():.4f}")
        print(f"    entropy_loss_weight={self.calculate_entropy_loss_weight(n_steps)}, commitment_loss_weight={self.commitment_loss_weight}")
        
        # 如果发现NaN，打印更多信息
        if not torch.isfinite(aux_loss).all():
            print("="*50)
            print("NaN DETECTED! Detailed info:")
            print(f"  entropy_aux_loss: {entropy_aux_loss}")
            print(f"  commit_loss: {commit_loss}")
            print(f"  entropy_loss_weight: {self.calculate_entropy_loss_weight(n_steps)}")
            print(f"  commitment_loss_weight: {self.commitment_loss_weight}")
            print("="*50)
            # 可选：抛出异常立即停止
            # raise ValueError("NaN detected in aux_loss!")

        return out, dict(indices=indices, aux_loss=aux_loss, codebook_usage=self.codebook_usage)
