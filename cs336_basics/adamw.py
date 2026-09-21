import torch
from torch import nn


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9,0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        if lr<0.0:
            raise ValueError(f"Invalid learning rate:{lr}")
        if eps<0.0:
            raise ValueError(f"Invalid epsilon value:{eps}")
        if not 0.0<=betas[0]<1.0:
            raise ValueError(f"Invalid beta parameter at index 0:{betas[0]}")
        if not 0.0<=betas[1]<1.0:
            raise ValueError(f"Invalid beta parameter at index 1:{betas[1]}")
        if weight_decay<0.0:
            raise ValueError(f"Invalid weight_decay value:{weight_decay}")

        defaults={
            "lr":lr,
            "betas":betas,
            "eps":eps,
            "weight_decay":weight_decay,
        }

        super().__init__(params,defaults)

    def step(self,closure=None):
        loss=None

        if closure is not None:
            with torch.enable_grad():
                loss=closure()

        for group in self.param_groups:
            lr=group["lr"]
            beta1,beta2=group["betas"]
            eps=group["eps"]
            weight_decay=group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad=param.grad

                state=self.state[param]

                if len(state)==0:
                    state["step"]=0
                    state["m"]=torch.zeros_like(param)
                    state["v"]=torch.zeros_like(param)

                state["step"]+=1

                m=state["m"]
                v=state["v"]

                m.mul_(beta1).add_(grad,alpha=1-beta1)
                v.mul_(beta2).addcmul_(grad,grad,value=1-beta2)

                step=state["step"]

                bias_correction1=1-beta1**step
                bias_correction2=1-beta2**step

                step_size=lr / bias_correction1

                denom=v.sqrt() / (bias_correction2**0.5)
                denom.add_(eps)

                with torch.no_grad():
                    param.addcdiv_(m,denom,value=-step_size)

                    # Decoupled weight decay
                    param.mul_(1-lr * weight_decay)

        return loss