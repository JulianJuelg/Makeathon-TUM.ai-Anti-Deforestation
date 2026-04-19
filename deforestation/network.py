"""Network definitions for deforestation segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """Small U-Net for AEF change segmentation."""

    def __init__(self, in_channels: int = 192, base_channels: int = 64):
        super().__init__()
        
        def conv_block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
            
        self.enc1 = conv_block(in_channels, base_channels)
        self.enc2 = conv_block(base_channels, base_channels * 2)
        self.enc3 = conv_block(base_channels * 2, base_channels * 4)
        
        self.pool = nn.MaxPool2d(2)
        
        self.dec3 = conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec2 = conv_block(base_channels * 2 + base_channels, base_channels)
        
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        
        self.final = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        d3 = self.up3(e3)
        diff_y = e2.size(2) - d3.size(2)
        diff_x = e2.size(3) - d3.size(3)
        if diff_y > 0 or diff_x > 0:
            d3 = F.pad(d3, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        d3 = torch.cat([e2, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        diff_y = e1.size(2) - d2.size(2)
        diff_x = e1.size(3) - d2.size(3)
        if diff_y > 0 or diff_x > 0:
            d2 = F.pad(d2, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        d2 = torch.cat([e1, d2], dim=1)
        d2 = self.dec2(d2)
        
        return self.final(d2)


class DilatedResidualBlock(nn.Module):
    """Resolution-preserving residual block with configurable dilation."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class DilatedChangeNet(nn.Module):
    """Small full-resolution network for AEF change segmentation."""

    def __init__(
        self,
        in_channels: int = 192,
        base_channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(base_channels, dilation=d)
            for d in dilations
        )
        self.mix = nn.Sequential(
            nn.Conv2d(base_channels * (len(dilations) + 1), base_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [self.stem(x)]
        h = features[0]
        for block in self.blocks:
            h = block(h)
            features.append(h)
        return self.final(self.mix(torch.cat(features, dim=1)))


class WindowTransformerBlock(nn.Module):
    """Transformer block with explicit attention to avoid SDPA CUDA issues."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        h = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = x + self.drop(self.proj(h))
        return x + self.mlp(self.norm2(x))


class WindowViTChangeNet(nn.Module):
    """Windowed ViT-style segmentation network that supports full-tile inference."""

    def __init__(
        self,
        in_channels: int = 192,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        window_size: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.pos_conv = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        )
        self.encoder = nn.Sequential(*[
            WindowTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.final = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def _window_partition(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2:]
        x = x.view(b, c, hp // ws, ws, wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.view(-1, ws * ws, c), hp, wp

    def _window_reverse(self, windows: torch.Tensor, b: int, c: int, hp: int, wp: int) -> torch.Tensor:
        ws = self.window_size
        x = windows.view(b, hp // ws, wp // ws, ws, ws, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, c, hp, wp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0, w0 = x.shape[-2:]
        x = self.stem(x)
        x = x + self.pos_conv(x)
        b, c = x.shape[:2]
        windows, hp, wp = self._window_partition(x)
        windows = self.encoder(windows)
        x = self._window_reverse(windows, b, c, hp, wp)
        x = x[:, :, :h0, :w0]
        x = self.refine(x)
        return self.final(x)
