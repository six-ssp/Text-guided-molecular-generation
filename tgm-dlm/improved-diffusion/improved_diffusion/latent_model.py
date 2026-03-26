import torch
import torch.nn as nn

from .nn import SiLU, linear, timestep_embedding


class LatentConditionedMLP(nn.Module):
    def __init__(
        self,
        latent_dim,
        model_channels=256,
        hidden_size=512,
        text_dim=768,
        dropout=0.1,
        text_fusion="pooled",
        text_attn_heads=8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.model_channels = model_channels
        self.text_fusion = text_fusion
        if self.text_fusion not in {"pooled", "crossattn"}:
            raise ValueError(f"unsupported text_fusion: {self.text_fusion}")

        self.time_embed = nn.Sequential(
            linear(model_channels, hidden_size),
            SiLU(),
            linear(hidden_size, hidden_size),
        )
        self.latent_proj = nn.Sequential(
            linear(latent_dim, hidden_size),
            SiLU(),
        )
        self.text_proj = nn.Sequential(
            linear(text_dim, hidden_size),
            SiLU(),
        )
        if self.text_fusion == "crossattn":
            if hidden_size % text_attn_heads != 0:
                raise ValueError(
                    f"hidden_size ({hidden_size}) must be divisible by text_attn_heads ({text_attn_heads})"
                )
            self.text_token_proj = nn.Sequential(
                linear(text_dim, hidden_size),
                SiLU(),
            )
            self.query_proj = nn.Sequential(
                linear(hidden_size * 2, hidden_size),
                SiLU(),
            )
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=text_attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.text_gate = nn.Sequential(
                linear(hidden_size * 2, hidden_size),
                SiLU(),
                linear(hidden_size, hidden_size),
                nn.Sigmoid(),
            )
        self.fuse = nn.Sequential(
            linear(hidden_size * 3, hidden_size),
            SiLU(),
            nn.Dropout(dropout),
            linear(hidden_size, hidden_size),
            SiLU(),
            linear(hidden_size, latent_dim),
        )

    def _pool_text(self, desc_state, desc_mask):
        if desc_mask is None:
            return desc_state.mean(dim=1)
        mask = desc_mask.to(dtype=desc_state.dtype).unsqueeze(-1)
        masked_state = desc_state * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        return masked_state.sum(dim=1) / denom

    def _fuse_text(self, desc_state, desc_mask, latent_hidden, time_hidden):
        desc_state = desc_state.to(dtype=latent_hidden.dtype)
        pooled_text = self._pool_text(desc_state, desc_mask)
        pooled_hidden = self.text_proj(pooled_text)
        if self.text_fusion == "pooled":
            return pooled_hidden

        token_hidden = self.text_token_proj(desc_state)
        query = self.query_proj(torch.cat([latent_hidden, time_hidden], dim=-1)).unsqueeze(1)
        key_padding_mask = None
        if desc_mask is not None:
            key_padding_mask = ~desc_mask.bool()
        attn_hidden, _ = self.cross_attn(
            query,
            token_hidden,
            token_hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attn_hidden = attn_hidden.squeeze(1)

        gate = self.text_gate(torch.cat([latent_hidden, time_hidden], dim=-1))
        return pooled_hidden + gate * attn_hidden

    def forward(self, x, timesteps, desc_state, desc_mask, **kwargs):
        if x.dim() != 2:
            x = x.view(x.size(0), -1)
        latent_hidden = self.latent_proj(x)
        time_hidden = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        text_hidden = self._fuse_text(desc_state, desc_mask, latent_hidden, time_hidden)
        hidden = torch.cat([latent_hidden, text_hidden, time_hidden], dim=-1)
        return self.fuse(hidden)
