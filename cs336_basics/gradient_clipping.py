import torch

def gradient_clipping(parameters,max_l2_norm,eps=1e-6):
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
    
    if not grads:
        return
    
    all_grads=torch.cat(grads)
    l2_norm=torch.norm(all_grads,p=2)

    if l2_norm>max_l2_norm:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.data*=max_l2_norm/(l2_norm+eps)