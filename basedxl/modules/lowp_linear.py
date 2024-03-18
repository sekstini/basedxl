import torch
import torch.nn as nn

from basedxl.triton.cascaded_lowp_matmul import cascaded_lowp_matmul


class BasedXLLowPLinear(nn.Module):
    def __init__(self, submodule: nn.Linear):
        super().__init__()
        self.in_features = submodule.in_features
        self.out_features = submodule.out_features
        self.weight = submodule.weight.T.contiguous().clone()
        self.bias = submodule.bias.clone() if submodule.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cascaded_lowp_matmul(x, self.weight)
        # TODO: Remove when we add bias support to the kernel
        if self.bias is not None:
            x = x + self.bias
        return x
