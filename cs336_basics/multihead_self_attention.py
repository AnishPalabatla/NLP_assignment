import torch
import torch.nn as nn
from einops import einsum, rearrange
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.linear import Linear


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # (h dkv, d_model)
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x):
        # Project input x into query, key, and value
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape Q,K,V for multi-head attention
        Q = rearrange(Q, '... l (h k) -> ... h l k', h=self.num_heads)
        K = rearrange(K, '... l (h k) -> ... h l k', h=self.num_heads)
        V = rearrange(V, '... l (h k) -> ... h l k', h=self.num_heads)

        # scaled dot-product attention
        seq_len = x.size(-2)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        causal_mask = ~mask  # upper triangular mask
        scaled_attention = scaled_dot_product_attention(Q, K, V, causal_mask)

        # Reshape output back to original dimensions
        output = rearrange(scaled_attention, '... h l k -> ... l (h k)')
        output = self.output_proj(output)

        return output.contiguous()


class MultiheadSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta

        # (h dkv, d_model)
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        # build once, reuse every forward() call instead of reconstructing per call
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)

    def forward(self, x, token_positions=None):
        # Project input x into query, key, and value
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape Q,K,V for multi-head attention
        Q = rearrange(Q, '... l (h k) -> ... h l k', h=self.num_heads)
        K = rearrange(K, '... l (h k) -> ... h l k', h=self.num_heads)
        V = rearrange(V, '... l (h k) -> ... h l k', h=self.num_heads)

        # Apply RoPE to Q and K
        if token_positions is not None:
            # token_positions is (..., seq_len) without a head dim, but Q/K are
            # (..., h, seq_len, k) after rearrange, so insert a broadcastable head dim
            # right before the seq_len axis.
            token_positions = token_positions.unsqueeze(-2)
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        # head-wise scaled dot-product attention
        seq_len = x.size(-2)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        causal_mask = ~mask  # upper triangular mask
        scaled_attention = scaled_dot_product_attention(Q, K, V, causal_mask)

        # Reshape output back to original dimensions
        output = rearrange(scaled_attention, '... h l k -> ... l (h k)')
        output = self.output_proj(output)

        return output.contiguous()