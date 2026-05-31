"""
MSFA-Net Lite: Lightweight Multi-Scale Feature Attention Network.

A simplified variant of MSFA-Net for efficient image dehazing.

Simplifications vs. full MSFA-Net:
    - 2 scales (full + ½) instead of 3 (full + ½ + ¼)
    - Simple residual conv blocks instead of DenseBlocks
    - 1 FA block per level instead of 2
    - 32 channels default instead of 64

Architecture:
    Input I(x)  ──────────────────────────────────┐ (global residual)
      ↓ [Conv 3→CH]                               │
    ┌── Encoder ───────────────────────────┐       │
    │ Scale 0: ConvBlock + GN + FA  ─skip─┐│       │
    │   ↓ stride-2 conv                   ││       │
    │ Scale 1: ConvBlock + GN + FA        ││       │
    └─────────────────────────────────────┘││       │
      ↓ upsample + conv                   ││       │
    ┌── Decoder ───────────────────────────┐│       │
    │ Scale 0: cat(skip) → ConvBlock+GN+FA ←┘      │
    └──────────────────────────────────────┘        │
      ↓ [Conv CH→3]                                 │
    clean = I(x) + α·residual  ←────────────────────┘
    clamp [0, 1]

References:
    Dong et al. (2020). MSBDN. CVPR.
    Qin et al. (2020). FFA-Net. AAAI.

Output: (B, 3, H, W) dehazed image in [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_CHANNELS = 64
DEFAULT_NUM_GROUPS = 8


# ── Reflection-padded convolution ────────────────────────────────────────────

class ReflPadConv2d(nn.Module):
    """Conv2d with reflection padding."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=0)

    def forward(self, x):
        return self.conv(self.pad(x))


# ── Channel Attention ────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.ca(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


# ── Pixel Attention ──────────────────────────────────────────────────────────

class PixelAttention(nn.Module):
    """Spatial attention via 1×1 conv."""

    def __init__(self, channels: int):
        super().__init__()
        mid = max(1, channels // 4)
        self.pa = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.pa(x)

# ── Feature Attention ────────────────────────────────────────────────────────

class FeatureAttention(nn.Module):
    """CA → PA with residual."""

    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.pa = PixelAttention(channels)

    def forward(self, x):
        return x + self.pa(self.ca(x))


# ── Simple Residual Conv Block ───────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Simple 2-layer residual conv block with GroupNorm.

        out = x + GN(ReLU(GN(Conv(Conv(x)))))

    Much simpler than DenseBlock — no concatenation, fewer params.
    """

    def __init__(self, channels: int, num_groups: int = DEFAULT_NUM_GROUPS):
        super().__init__()
        self.block = nn.Sequential(
            ReflPadConv2d(channels, channels, kernel_size=3),
            nn.GroupNorm(num_groups, channels),
            nn.ReLU(inplace=True),
            ReflPadConv2d(channels, channels, kernel_size=3),
            nn.GroupNorm(num_groups, channels),
        )

    def forward(self, x):
        return x + self.block(x)  # residual


# ── MSFA-Net Lite ────────────────────────────────────────────────────────────

class MSFANetLite(nn.Module):
    """
    Lightweight Multi-Scale Feature Attention Network.

    2-scale encoder-decoder with simple conv blocks and single FA per level.

    Args:
        channels: Backbone feature width (default 32).
    """

    def __init__(self, channels: int = DEFAULT_CHANNELS):
        super().__init__()
        ch = channels
        ng = max(1, ch // 8)  # num_groups for GroupNorm

        # ── Input Projection ─────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            ReflPadConv2d(3, ch, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        # ── Encoder ──────────────────────────────────────────────────────────
        # Scale 0 (full resolution)
        self.enc0 = nn.Sequential(
            ConvBlock(ch, ng),
            nn.GroupNorm(ng, ch),
            FeatureAttention(ch),
        )
        self.down = nn.Sequential(
            ReflPadConv2d(ch, ch, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        # Scale 1 (½ resolution)
        self.enc1 = nn.Sequential(
            ConvBlock(ch, ng),
            nn.GroupNorm(ng, ch),
            FeatureAttention(ch),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up_conv = ReflPadConv2d(ch, ch, kernel_size=3)
        # Skip fusion: 2·CH → CH
        self.skip_fuse = nn.Sequential(
            nn.Conv2d(ch * 2, ch, kernel_size=1),
            nn.GroupNorm(ng, ch),
            nn.ReLU(inplace=True),
        )
        self.dec0 = nn.Sequential(
            ConvBlock(ch, ng),
            nn.GroupNorm(ng, ch),
            FeatureAttention(ch),
        )

        # ── Output Projection ────────────────────────────────────────────────
        self.output_proj = ReflPadConv2d(ch, 3, kernel_size=3)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        # ── Initialization ───────────────────────────────────────────────────
        self._initialize_weights()
        nn.init.zeros_(self.output_proj.conv.weight)
        nn.init.zeros_(self.output_proj.conv.bias)

    def forward(self, x):
        """
        Args:
            x: Hazy input (B, 3, H, W) in [0, 1].
        Returns:
            Dehazed image (B, 3, H, W) in [0, 1].
        """
        f0 = self.input_proj(x)              # (B, CH, H, W)

        # Encoder
        e0 = self.enc0(f0)                   # (B, CH, H, W)    — skip
        f1 = self.down(e0)                   # (B, CH, H/2, W/2)
        e1 = self.enc1(f1)                   # (B, CH, H/2, W/2)

        # Decoder
        u0 = F.interpolate(e1, scale_factor=2, mode='bilinear',
                           align_corners=False)
        u0 = F.relu(self.up_conv(u0), inplace=True)
        # Handle odd sizes
        if u0.shape[2:] != e0.shape[2:]:
            u0 = F.interpolate(u0, size=e0.shape[2:], mode='bilinear',
                               align_corners=False)
        fused = self.skip_fuse(torch.cat([u0, e0], dim=1))
        d0 = self.dec0(fused)                # (B, CH, H, W)

        # Global residual
        residual = self.output_proj(d0)
        clean = x + self.residual_scale * residual

        return torch.clamp(clean, 0.0, 1.0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = MSFANetLite(channels=32)
    total = sum(p.numel() for p in model.parameters())
    dummy = torch.rand(1, 3, 256, 256)
    out = model(dummy)
    print(f"MSFA-Net Lite (ch=32)")
    print(f"  Params:  {total:,}")
    print(f"  Input:   {dummy.shape}")
    print(f"  Output:  {out.shape}, [{out.min():.4f}, {out.max():.4f}]")
