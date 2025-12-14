import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RegionConstrainedAttnProcessor:
    r"""
    Attention processor for Region-Constrained Cross-Attention (RCCA).
    """
    def __init__(self):
        pass

    def __call__(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        rcca_masks: Optional[torch.Tensor] = None,  # Custom argument for RCCA
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # --- RCCA Injection Point ---
        if rcca_masks is not None:
            # rcca_masks should be a dictionary mapping resolution (H*W) to mask tensor
            # The mask tensor should have shape compatible with attention_probs
            # expected attention_probs shape: (batch_size * heads, query_len, key_len)
            
            # The query_len corresponds to the spatial resolution (H*W)
            current_resolution = sequence_length
            
            if current_resolution in rcca_masks:
                 # Fetch mask: (batch_size, 1, query_len, key_len) or similar
                 # We need to broadcast to (batch_size * heads, query_len, key_len)
                 mask = rcca_masks[current_resolution]
                 
                 # Assuming mask is (batch_size, query_len, key_len) or (batch_size, 1, query_len, key_len)
                 if mask.ndim == 3:
                     # (B, Q, K) -> (B, 1, Q, K)
                     mask = mask.unsqueeze(1)
                 
                 # Expand to match heads: (B, H, Q, K) -> (B*H, Q, K)
                 mask = mask.repeat(1, attn.heads, 1, 1)
                 mask = mask.view(batch_size * attn.heads, current_resolution, -1)
                 
                 # Apply mask
                 attention_probs = attention_probs + mask
        # ----------------------------

        attention_probs = attention_probs.softmax(dim=-1)
        attention_probs = attention_probs.to(value.dtype)

        if attention_mask is not None:
             # This seems redundant if pre-applied in get_attention_scores, but diffusers does it sometimes?
             # Actually get_attention_scores usually returns raw scores (Q*K) + bias.
             # So softmax is applied after.
             pass

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states
