import torch
from einops import rearrange, reduce

def cross_entropy(logits,targets):
    max_logits=reduce(logits,'b ... v -> b ... 1','max')
    logits=logits-max_logits
    exp_logits=torch.exp(logits)
    sum_exp_logits=reduce(exp_logits, 'b ... v -> b ...','sum')
    logsumexp=torch.mean(torch.log(sum_exp_logits))

    logits=rearrange(logits,'b ... v -> (b ...) v')
    targets=rearrange(targets,'b ... -> (b ...)')
    targets=logits[torch.arange(targets.shape[0]),targets]
    loss=logsumexp-torch.mean(targets)

    return loss