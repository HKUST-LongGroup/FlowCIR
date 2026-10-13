"""
the flow matching network, from MAR.
"""



import math
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection


def modulate(x, shift, scale):
    return x * (1+scale) + shift
 

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: [N,1], float in [0,1]
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t * 1000.0  # scale t from [0,1] to [0,1000]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    
class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h
    
class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x





class DeepFlowMatchingRegressorNet(nn.Module):
    """
    A regressor network without time conditioning.
    Supports conditioning: forward(x, cond)
    - x:    [N, in_channels]
    - cond: [N, in_channels] (same shape as x)
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        # Project input x -> hidden
        self.input_proj = nn.Linear(in_channels, model_channels)

        # Project condition cond -> hidden
        self.cond_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, cond):
        """
        Apply the model to an input batch with condition.
        :param x:    [N, in_channels]
        :param cond: [N, in_channels] (same shape as x)
        :return: regression output [N, out_channels]
        """
        if cond.shape != x.shape:
            raise ValueError(f"cond must have same shape as x, got x={x.shape}, cond={cond.shape}")

        # Hidden states
        x_h = self.input_proj(x)        # [N, model_channels]

        # Conditioning vector: only use cond embedding
        y = self.cond_proj(cond)        # [N, model_channels]

        # Residual blocks with adaLN modulated by y
        for block in self.res_blocks:
            x_h = block(x_h, y)

        output = self.final_layer(x_h, y)   # [N, out_channels]
        return output





class DeepFlowMatchingNet_context(nn.Module):
    """
    DeepFlowMatchingNet (v1) with optional `context` concatenated to x.

    forward(x, cond, t, context=None)
    - x:       [N, in_channels]
    - context: [N, context_channels] (optional)
    - cond:    [N, cond_channels]  (usually same as in_channels in your current setup)
    - t:       [N] or [N,1] float in [0,1]
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        context_channels: int = 768,     # NEW: how many dims of context to concat
        cond_channels: int = None,     # NEW(optional): allow cond dim differ from x dim
    ):
        super().__init__()

        self.in_channels = in_channels
        self.context_channels = int(context_channels)
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        if cond_channels is None:
            cond_channels = in_channels
        self.cond_channels = int(cond_channels)

        self.time_embed = TimestepEmbedder(model_channels)

        # NEW: input_proj now expects x (+ context if provided)
        input_in_dim = in_channels + self.context_channels
        self.input_proj = nn.Linear(input_in_dim, model_channels)

        # cond projection can be different dim if you want; default = in_channels
        self.cond_proj = nn.Linear(self.cond_channels, model_channels)

        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor, context: torch.Tensor = None):
        """
        :param x:       [N, in_channels]
        :param cond:    [N, cond_channels]
        :param t:       [N] or [N,1]
        :param context: [N, context_channels] or None
        """
        if x.ndim != 2:
            raise ValueError(f"x must be [N, D], got {x.shape}")
        if cond.ndim != 2:
            raise ValueError(f"cond must be [N, D], got {cond.shape}")
        if cond.shape[0] != x.shape[0]:
            raise ValueError(f"batch mismatch: x={x.shape}, cond={cond.shape}")
        if cond.shape[1] != self.cond_channels:
            raise ValueError(f"cond dim mismatch: expected {self.cond_channels}, got {cond.shape[1]}")

        # NEW: concat context if provided
        if context is not None:
            if context.ndim != 2:
                raise ValueError(f"context must be [N, Dctx], got {context.shape}")
            if context.shape[0] != x.shape[0]:
                raise ValueError(f"batch mismatch: x={x.shape}, context={context.shape}")
            if context.shape[1] != self.context_channels:
                raise ValueError(f"context dim mismatch: expected {self.context_channels}, got {context.shape[1]}")
            x_in = torch.cat([x, context], dim=-1)   # [N, in_channels + context_channels]
        else:
            if self.context_channels != 0:
                # you declared context_channels>0 but didn't pass context
                # allow it, but be explicit:
                x_in = x
            else:
                x_in = x

        # Hidden states
        x_h = self.input_proj(x_in)  # [N, C]

        # Conditioning vector: y = t_emb + cond_emb
        t_emb = self.time_embed(t)       # [N, C]
        cond_emb = self.cond_proj(cond)  # [N, C]
        y = t_emb + cond_emb             # [N, C]

        # Residual blocks modulated by y
        for block in self.res_blocks:
            x_h = block(x_h, y)

        velocity = self.final_layer(x_h, y)  # [N, out_channels]
        return velocity


class DeepFlowMatchingNet(nn.Module):
    """
    The MLP for Velocity Field.
    Now supports conditioning: forward(x, cond, t)
    - x:    [N, in_channels]
    - cond: [N, in_channels] (same shape as x)
    - t:    [N] or [N, 1] float in [0, 1]
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.time_embed = TimestepEmbedder(model_channels)

        # Project input x -> hidden
        self.input_proj = nn.Linear(in_channels, model_channels)

        # Project condition cond -> hidden, then add to timestep embedding (FM-style conditioning)
        self.cond_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, cond, t):
        """
        Apply the model to an input batch with condition.
        :param x:    [N, in_channels]
        :param cond: [N, in_channels] (same shape as x)
        :param t:    [N] or [N, 1] float in [0, 1]
        :return: velocity [N, out_channels]
        """
        if cond.shape != x.shape:
            raise ValueError(f"cond must have same shape as x, got x={x.shape}, cond={cond.shape}")

        # Hidden states
        x_h = self.input_proj(x)                 # [N, model_channels]

        # Conditioning vector (FM-style): y = t_emb + cond_emb
        t_emb = self.time_embed(t)               # [N, model_channels]
        cond_emb = self.cond_proj(cond)          # [N, model_channels]
        y = t_emb + cond_emb                     # [N, model_channels]

        # Residual blocks with adaLN modulated by y
        for block in self.res_blocks:
            x_h = block(x_h, y)

        velocity = self.final_layer(x_h, y)      # [N, out_channels]
        return velocity




class DeepFlowMatchingNet_v2(nn.Module):
    """
    DeepFlowMatchingNet v2: supports TWO conditions injected FM-style.

    forward(x, cond_text, cond_img, t)
    - x:         [N, in_channels]
    - cond_text: [N, in_channels] (same shape as x)
    - cond_img:  [N, in_channels] (same shape as x)
    - t:         [N] or [N, 1] float in [0, 1]
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.time_embed = TimestepEmbedder(model_channels)

        # Project input x -> hidden
        self.input_proj = nn.Linear(in_channels, model_channels)

        # Project two conditions -> hidden
        self.cond_text_proj = nn.Linear(in_channels, model_channels)
        self.cond_img_proj = nn.Linear(in_channels, model_channels)

        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, cond_text, cond_img, t):
        """
        :param x:         [N, in_channels]
        :param cond_text: [N, in_channels]
        :param cond_img:  [N, in_channels]
        :param t:         [N] or [N, 1] float in [0, 1]
        :return: velocity [N, out_channels]
        """
        if cond_text.shape != x.shape:
            raise ValueError(f"cond_text must have same shape as x, got x={x.shape}, cond_text={cond_text.shape}")
        if cond_img.shape != x.shape:
            raise ValueError(f"cond_img must have same shape as x, got x={x.shape}, cond_img={cond_img.shape}")

        # Hidden states
        x_h = self.input_proj(x)  # [N, model_channels]

        # Conditioning vector: y = t_emb + cond_text_emb + cond_img_emb
        t_emb = self.time_embed(t)                 # [N, model_channels]
        cond_text_emb = self.cond_text_proj(cond_text)  # [N, model_channels]
        cond_img_emb = self.cond_img_proj(cond_img)     # [N, model_channels]
        y = t_emb + cond_text_emb + cond_img_emb   # [N, model_channels]

        # Residual blocks with adaLN modulated by y
        for block in self.res_blocks:
            x_h = block(x_h, y)

        velocity = self.final_layer(x_h, y)        # [N, out_channels]
        return velocity




class DeepFlowMatchingNet_v3(nn.Module):
    """
    DeepFlowMatchingNet v3: concat + MLP fusion for multi-condition injection.

    forward(x, cond_text, cond_img, t)
    - x:         [N, in_channels]
    - cond_text: [N, in_channels] (same shape as x)
    - cond_img:  [N, in_channels] (same shape as x)
    - t:         [N] or [N, 1] float in [0, 1]

    Fusion:
      t_emb = time_embed(t)                      -> [N, C]
      c_t   = cond_text_proj(cond_text)          -> [N, C]
      c_i   = cond_img_proj(cond_img)            -> [N, C]
      y     = fuse_mlp([t_emb, c_t, c_i])        -> [N, C]
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        fuse_hidden_mult: int = 4,   # MLP hidden = fuse_hidden_mult * model_channels
        fuse_dropout: float = 0.0,   # keep 0.0 for DiT-style stability unless needed
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.time_embed = TimestepEmbedder(model_channels)

        # Project input x -> hidden
        self.input_proj = nn.Linear(in_channels, model_channels)

        # Project conditions -> hidden
        self.cond_text_proj = nn.Linear(in_channels, model_channels)
        self.cond_img_proj = nn.Linear(in_channels, model_channels)

        # Concat + MLP fusion
        fuse_in = 3 * model_channels
        fuse_hidden = fuse_hidden_mult * model_channels
        self.fuse_mlp = nn.Sequential(
            nn.LayerNorm(fuse_in, eps=1e-6),
            nn.Linear(fuse_in, fuse_hidden, bias=True),
            nn.SiLU(),
            nn.Dropout(p=fuse_dropout),
            nn.Linear(fuse_hidden, model_channels, bias=True),
        )

        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Optional: make fusion start near "sum-like" by zeroing last layer,
        # then y starts near 0; BUT since y modulates adaLN and those are zeroed,
        # training is usually still stable either way. We'll keep default init.

        # Zero-out adaLN modulation layers in ResBlocks (DiT trick)
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers (DiT trick)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, cond_text: torch.Tensor, cond_img: torch.Tensor, t: torch.Tensor):
        if cond_text.shape != x.shape:
            raise ValueError(f"cond_text must have same shape as x, got x={x.shape}, cond_text={cond_text.shape}")
        if cond_img.shape != x.shape:
            raise ValueError(f"cond_img must have same shape as x, got x={x.shape}, cond_img={cond_img.shape}")

        # Hidden states
        x_h = self.input_proj(x)  # [N, model_channels]

        # Embeddings
        t_emb = self.time_embed(t)                 # [N, model_channels]
        c_t = self.cond_text_proj(cond_text)       # [N, model_channels]
        c_i = self.cond_img_proj(cond_img)         # [N, model_channels]

        # Concat + fuse
        y = self.fuse_mlp(torch.cat([t_emb, c_t, c_i], dim=-1))  # [N, model_channels]

        # Residual blocks modulated by y
        for block in self.res_blocks:
            x_h = block(x_h, y)

        velocity = self.final_layer(x_h, y)  # [N, out_channels]
        return velocity

class DeepFlowMatchingNet_v1_2(nn.Module):
    """
    The MLP for Velocity Field.
    Conditioning is combined via concat + fuse:
      y = Fuse([t_emb, cond_emb])

    forward(x, cond, t)
    - x:    [N, in_channels]
    - cond: [N, in_channels] (same shape as x)
    - t:    [N] or [N, 1] float in [0, 1]
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.time_embed = TimestepEmbedder(model_channels)

        # Project input x -> hidden
        self.input_proj = nn.Linear(in_channels, model_channels)

        # Project condition cond -> hidden
        self.cond_proj = nn.Linear(in_channels, model_channels)

        # Concat + fuse: [t_emb, cond_emb] -> y
        self.fuse_mlp = nn.Sequential(
            nn.Linear(2 * model_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )

        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # (Optional but usually helpful) start from "weak conditioning":
        # make fuse output initially ~0 so adaLN mod starts near identity.
        nn.init.constant_(self.fuse_mlp[-1].weight, 0)
        nn.init.constant_(self.fuse_mlp[-1].bias, 0)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, cond, t):
        """
        :param x:    [N, in_channels]
        :param cond: [N, in_channels] (same shape as x)
        :param t:    [N] or [N, 1] float in [0, 1]
        :return: velocity [N, out_channels]
        """
        if cond.shape != x.shape:
            raise ValueError(f"cond must have same shape as x, got x={x.shape}, cond={cond.shape}")

        # Hidden states
        x_h = self.input_proj(x)  # [N, model_channels]

        # Concat + fuse conditioning
        t_emb = self.time_embed(t)      # [N, model_channels]
        cond_emb = self.cond_proj(cond) # [N, model_channels]
        y = self.fuse_mlp(torch.cat([t_emb, cond_emb], dim=-1))  # [N, model_channels]

        # Residual blocks with adaLN modulated by y
        for block in self.res_blocks:
            x_h = block(x_h, y)

        velocity = self.final_layer(x_h, y)  # [N, out_channels]
        return velocity







