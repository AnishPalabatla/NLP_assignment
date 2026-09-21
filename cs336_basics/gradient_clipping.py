import torch
from collections.abc import Iterable

def gradient_clipping(
    parameters:Iterable[torch.nn.Parameter],
    max_l2_norm:float,
) -> None:
    parameters=list(parameters)

    total_norm_squared=sum(
        torch.sum(param.grad**2)
        for param in parameters
        if param.grad is not None
    )

    total_norm=torch.sqrt(total_norm_squared)
    if total_norm>max_l2_norm:
        scale=max_l2_norm/(total_norm+1e-6)
        for param in parameters:
            if param.grad is not None:
                param.grad.mul_(scale)