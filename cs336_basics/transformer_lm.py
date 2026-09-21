import torch
import torch.nn as nn
from cs336_basics.embedding import Embedding
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear
from cs336_basics.softmax import softmax

class TransformerLM(nn.Module):
    def __init__(self,vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta):
        super().__init__()
        self.num_layers=num_layers
        self.token_embeddings=Embedding(vocab_size,d_model)
        self.layers=nn.ModuleList([
            TransformerBlock(d_model,num_heads,d_ff,context_length,rope_theta) for i in range(num_layers)
        ])
        self.ln_final=RMSNorm(d_model)
        self.lm_head=Linear(d_model,vocab_size)

    def forward(self,x):
        x=self.token_embeddings(x)
        for layer in self.layers:
            x=layer(x)
        x=self.ln_final(x)
        logits=self.lm_head(x)
        return logits