import torch
import torch.nn as nn
from einops import repeat

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device=None):
        super().__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_len=max_seq_len
        self.device=device

        angle=1/(theta**(torch.arange(0,d_k,2,device=device).float()/d_k))
        position=torch.arange(max_seq_len,device=device)
        matrix=torch.outer(position,angle)

        self.register_buffer('sin',torch.sin(matrix),persistent=False)
        self.register_buffer('cos',torch.cos(matrix),persistent=False)

    def forward(self,x,token_positions=None):
        if token_positions is None:
            positions=torch.arange(x.size(-2),device=x.device)
            dims=[f'b{i}' for i in range(x.ndim-2)]
            shapes_dict={dim:shape for dim,shape in zip(dims,x.shape[:-2])}
            token_positions=repeat(positions,f"l -> {' '.join(dims)} l",**shapes_dict)

        sin=self.sin.to(x.device)[token_positions]
        cos=self.cos.to(x.device)[token_positions]

        x_rotary=x.clone()
        x_rotary[...,0::2]=x[...,0::2]*cos-x[...,1::2]*sin
        x_rotary[...,1::2]=x[...,0::2]*sin+x[...,1::2]*cos

        return x_rotary