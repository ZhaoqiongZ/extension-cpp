import torch
from torch import Tensor
__all__ = ["mymuladd"]

def mymuladd(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.sycl_extension.mymuladd.default(a, b, c)