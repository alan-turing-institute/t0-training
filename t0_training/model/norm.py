import torch
import torch.nn as nn

# https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/nn/layer_norm.py#L210

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps # to avoid division by zero
        self.weight = nn.Parameter(torch.ones(d_model)) # learnable scaling parameter of shape (d_model,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        # x has shape (B, T, d_model) where B = batch size, T = sequence length (n. tokens), d_model = hidden dimension

        # Compute the variance in fp32 regardless of input dtype: summing ~d_model
        # squared elements in bf16 loses precision. Cast back to x's dtype after.
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt() # calculate RMS along the last dim (d_model)
        return ((x / rms).to(dtype)) * self.weight
