import math

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetBlockType(Enum):
    UP = 0
    DOWN = 1


"""
Swish Activation Function.
"""
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


"""
Spatially-Adaptive Normalization: (SPADE).
"""
class SPADE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.y_scale = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1))
        self.y_shift = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1))

    def forward(self, x, style):
        _, _, D, H, W = x.shape

        y_scale = self.y_scale(style)
        y_scale = F.interpolate(
            y_scale,
            size=(D, H, W),
            mode="area")

        y_shift = self.y_shift(style)
        y_shift = F.interpolate(
            y_shift,
            size=(D, H, W),
            mode="area")

        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_std = torch.std(x, dim=1, keepdim=True)
        
        x = y_scale * ((x - x_mean) / x_std) + y_shift
        return x


"""
Adaptive Group Normalization: (AdaGN).
"""
class AdaGN(nn.Module):
    def __init__(self, emb_dim, out_dim, groups=32):
        super().__init__()

        self.y_scale = nn.Linear(emb_dim, out_dim)
        self.y_shift = nn.Linear(emb_dim, out_dim)

        self.group_norm = nn.GroupNorm(groups, out_dim)

    def forward(self, x, emb):
        x_gn = self.group_norm(x)

        combined_y_scale = None
        combined_y_shift = None
        emb_permuted = emb.permute(1, 0, 2)

        for emb_each in emb_permuted:
            y_scale = self.y_scale(emb_each)
            y_scale = y_scale[:, :, None, None, None]
            if combined_y_scale is None:
                combined_y_scale = y_scale
            else:
                combined_y_scale = torch.cat(
                    (combined_y_scale, y_scale),
                    dim=2
                )

            y_shift = self.y_scale(emb_each)
            y_shift = y_shift[:, :, None, None, None]

            if combined_y_shift is None:
                combined_y_shift = y_shift
            else:
                combined_y_shift = torch.cat(
                    (combined_y_shift, y_shift),
                    dim=2
                )
        x = combined_y_scale * x_gn + combined_y_shift
        return x


"""
Attention Block. Similar to transformer's mulit-head attention block.
Modified to allow for SpatioTemporal input i.e sequential frames in a video.
"""
class AttentionBlock(nn.Module):
    def __init__(self, channels, heads=1, d_k=None, groups=32):
        super().__init__()

        # Number of dimensions in each head.
        if d_k is None:
            d_k = channels
        
        # Normalization Layer.
        self.norm = nn.GroupNorm(groups, channels)

        # Projections for query(q), key(k) and values(v).
        self.projection = nn.Linear(channels, heads * d_k * 3)

        # Linear Layer for final transformation.
        self.output = nn.Linear(heads * d_k, channels)

        # Scale for dot-product attention.
        self.scale = d_k ** -0.5

        self.heads = heads
        self.d_k = d_k
    
    def forward(self, x, t=None):
        # Not used, but kept in the arguments because for the attention
        # layer function signature to match with ResidualBlock.
        _ = t
        
        # Batch Size (N), Channel (C), Frames (F), Height (H), Width (W)
        N, C, F, H, W = x.shape

        # Reshape to (N, seq, channels)
        x = x.view(N, C, -1).permute(0, 2, 1)

        # Get concatenated qkv and reshape to: (N, seq, heads, 3 * d_k)
        qkv = self.projection(x).view(N, -1, self.heads, 3 * self.d_k)
        
        # Split q, k, and v @ (N, seq, heads, d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Calculate scaled dot-product: QK^T / sqrt(d_k)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        
        # Softmax along sequence dimension.
        attn = attn.softmax(dim=1)

        # Multiply attn by v.
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        
        # Reshape to N, seq, heads * d_k.
        res = res.reshape(N, -1, self.heads * self.d_k)
        
        # Transform to N, seq, channels.
        res = self.output(res)

        # Add skip connection.
        res += x

        # Reshape to: N, C, H, W
        res = res.permute(0, 2, 1).view(N, C, F, H, W)

        return res


"""
Upsample using ConvTranspose3d.
"""
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=(1, 4, 4),
                stride=(1, 2, 2),
                padding=(0, 1, 1)),
            Swish(),)

    def forward(self, x, emb=None, lr_img=None):
        _ = emb
        _ = lr_img
        x = self.conv_layer(x)
        return x


"""
Downsample using Conv3d.
"""
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1)),
            Swish())

    def forward(self, x, emb=None, lr_img=None):
        _ = emb
        _ = lr_img
        x = self.conv_layer(x)
        return x


"""
Mapping Layer, similar to that from StyleGAN implementation.
Instead of passing Noise like with StyleGAN, pass low resolution image or mask.
"""
class MappingLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels=512):
        super().__init__()

        self.map_layer = nn.Sequential(
            nn.Conv3d(
                in_channels,
                hidden_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1)),
            Swish(),
            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1)),
            Swish(),
            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1)),
            Swish(),
            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1)),
            Swish(),
            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1))
        )
    
    def forward(self, x):
        x_map = self.map_layer(x)
        return x_map


"""
Time Embedding (Positional Sinusodial) + Conditional Info e.g one-hot encoding.
"""
class ConditionalEmbedding(nn.Module):
    def __init__(self, time_dim, label_dim=None):
        super().__init__()

        # Number of dimensions in the embedding.
        self.time_dim = time_dim
        self.label_dim = label_dim

        self.frame_layer = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            Swish(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        
        self.time_layer = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            Swish(),
            nn.Linear(self.time_dim, self.time_dim),
            Swish(),
            nn.Linear(self.time_dim, self.time_dim),
            Swish(),
            nn.Linear(self.time_dim, self.time_dim)
        )

        if self.label_dim is not None:
            self.label_layer = nn.Sequential(
                nn.Linear(self.label_dim, self.time_dim),
                Swish(),
                nn.Linear(self.time_dim, self.time_dim),
                Swish(),
                nn.Linear(self.time_dim, self.time_dim),
                Swish(),
                nn.Linear(self.time_dim, self.time_dim)
            )
        else:
            self.label_layer = None

    def forward(self, t, labels=None):
        # Sinusoidal Position embeddings.
        half_dim = self.time_dim // 2
        constant = math.log(10_000) / (half_dim - 1)

        arange_emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=t.device) * -constant
        )

        time_emb = t[:, None] * arange_emb[None, :]
        time_emb = torch.cat((time_emb.sin(), time_emb.cos()), dim=1)
        
        time_emb = self.time_layer(time_emb)
        time_emb = time_emb.unsqueeze(1)
        
        label_emb = None
        if self.label_layer is not None and labels is not None:
            labels = labels.permute(1, 0, 2)
            for frame_index, label in enumerate(labels):
                frame_count = torch.tensor([frame_index + 1], device=labels.device)

                frame_emb = frame_count[:, None] * arange_emb[None, :]
                frame_emb = torch.cat((frame_emb.sin(), frame_emb.cos()), dim=1)
                
                frame_emb = self.frame_layer(frame_emb)

                label_out = self.label_layer(label)
                label_out = label_out + frame_emb
                
                if label_emb is None:
                    label_emb = label_out.unsqueeze(1)
                else:
                    label_emb = torch.cat((label_emb, label_out.unsqueeze(1)), dim=1)
        else:
            label_emb = 0

        aggr_emb = label_emb + time_emb
        return aggr_emb



"""
Conditional Embedded Conv. Block.
"""
class UNet_ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            mapping_channels=None,
            use_activation=True,
            emb_dim=None,
            groups=32):
        super().__init__()
        
        conv_list = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1))
        ]
        if use_activation:
            conv_list.append(Swish())
        self.conv_layer = nn.Sequential(*conv_list)

        # Labels + Time Embedding.
        if emb_dim is not None:
            self.adagn = AdaGN(
                emb_dim,
                out_channels,
                groups=groups)

        # Low Resolution Embedding, 
        # Uses SPADE and not concatened like with other Super-Resolution models.
        if mapping_channels is not None:
            self.map_SPADE = SPADE(
                in_channels=mapping_channels,
                out_channels=out_channels)

    def forward(self, x, label_emb=None, cond_img=None):
        x = self.conv_layer(x)
        
        # Time and Label Embeddings.
        if label_emb is not None:
            x = self.adagn(x, label_emb)

        # Low Resolution Embeddings.
        if cond_img is not None:
            x = self.map_SPADE(x, cond_img)

        return x


"""
Residual Conv. Block.
"""
class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            mapping_channels=None,
            use_activation=True,
            emb_dim=None,
            groups=32):
        super().__init__()

        self.conv_block_1 = UNet_ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            mapping_channels=mapping_channels,
            use_activation=use_activation,
            emb_dim=emb_dim,
            groups=groups)
        self.conv_block_2 = UNet_ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            mapping_channels=mapping_channels,
            use_activation=use_activation,
            emb_dim=emb_dim,
            groups=groups)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 1, 1))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, label_emb=None, cond_img=None):
        init_x = x
        x = self.conv_block_1(
            x,
            label_emb,
            cond_img)
        x = self.conv_block_2(
            x,
            label_emb,
            cond_img)
        x = x + self.shortcut(init_x)
        return x


"""
U-Net Block.
"""
class UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_dim,
        mapping_channels=None,
        num_resnet_blocks=1,
        use_attn=True,
        num_heads=1,
        dim_per_head=None,
        groups=32,
        block_type=UNetBlockType.DOWN):
        super().__init__()

        hidden_channels = in_channels

        self.res_layers = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        for layer_count in range(num_resnet_blocks):
            self.res_layers.append(
                ResidualBlock(
                    in_channels=in_channels if layer_count == 0 else hidden_channels,
                    out_channels=hidden_channels,
                    mapping_channels=mapping_channels,
                    emb_dim=emb_dim))
            if use_attn:
                self.attn_layers.append(
                    AttentionBlock(
                        channels=hidden_channels,
                        heads=num_heads,
                        d_k=dim_per_head,
                        groups=groups))
            else:
                self.attn_layers.append(nn.Identity())

        if block_type == UNetBlockType.DOWN:
            self.out_layer = DownsampleBlock(
                in_channels=hidden_channels,
                out_channels=out_channels)
        elif block_type == UNetBlockType.UP:
            self.out_layer = UpsampleBlock(
                in_channels=hidden_channels,
                out_channels=out_channels)

    def forward(self, x, label_emb=None, cond_image=None):
        for res_layer, attn_layer in zip(
                self.res_layers,
                self.attn_layers):
            x = res_layer(x, label_emb, cond_image)
            x = attn_layer(x)
        x = self.out_layer(x, label_emb, cond_image)
        return x
