import torch
from torch import nn

def softmax(x,i):
    max_i=torch.max(x,dim=i,keepdim=True)[0]
    exp_x=torch.exp(x-max_i)
    x=exp_x/torch.sum(exp_x,dim=i,keepdim=True)
    return x

def scaled_dot_product_attention(query,key,value,mask=None):
    d_k=query.size(-1)
    scores=torch.matmul(query,key.transpose(-2,-1))/(d_k**0.5)
    if mask is not None:
        mask=mask.to(scores.device)
        scores=scores.masked_fill(~mask,float('-inf'))
    attn_weights=softmax(scores,-1)
    output=torch.matmul(attn_weights,value)
    return output