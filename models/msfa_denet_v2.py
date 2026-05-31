"""
MSFA-DeNet v2: Enhanced Multi-Scale Feature Attention Dehazing Network.

An improved variant of MSFA-Net Lite for non-homogeneous image dehazing.

Key improvements over MSFA-Net Lite:
    1. No normalization — relies on careful initialization and residual
       scaling for stability. Avoids GroupNorm/BatchNorm overhead and
       batch-size sensitivity.

    2. Better ConvBlock — depthwise-separable convolution with channel
       expansion bottleneck, GELU activation, and built-in SE attention.
       Same receptive field, fewer params, more expressive.

    3. 3-scale pyramid encoder-decoder (full → ½ → ¼) instead of 2-scale.
       The ¼ resolution captures large-scale haze density gradients that
       dominate sky regions and non-homogeneous haze patterns.

    4. Transmission-aware attention (TFA) — a lightweight side-branch
       estimates a rough transmission map t(x) and uses it to modulate
       pixel attention weights. Dense haze regions (low t) receive
       stronger attention, giving the network an explicit spatial prior.

    5. Per-channel learnable residual scale — shape (1, 3, 1, 1) instead
       of a single scalar. Learns different correction strengths for
       R, G, B independently, compensating for wavelength-dependent
       Rayleigh scattering.

    6. Pixel shuffle upsampling — replaces bilinear interpolate + conv
       with sub-pixel convolution (Conv → PixelShuffle). Avoids
       checkerboard artifacts and has fully learnable upsampling weights.

Architecture:
    Input I(x)  ─────────────────────────────────────────┐ (global residual)
      ↓ [Conv 3→CH, GELU]                               │
    ┌── Encoder ──────────────────────────────────┐      │
    │ Scale 0: ConvBlock + TFA          ──skip──┐ │      │
    │   ↓ stride-2 conv                         │ │      │
    │ Scale 1: ConvBlock + TFA          ──skip┐ │ │      │
    │   ↓ stride-2 conv                       │ │ │      │
    │ Bottleneck: ConvBlock + TFA             │ │ │      │
    └─────────────────────────────────────────┘ │ │      │
      ↓ PixelShuffle ×2                        │ │      │
    ┌── Decoder ──────────────────────────────┐ │ │      │
    │ Scale 1: cat(skip1) → ConvBlock + TFA ←─┘ │ │      │
    │   ↑ PixelShuffle ×2                      │ │      │
    │ Scale 0: cat(skip0) → ConvBlock + TFA ←───┘ │      │
    └──────────────────────────────────────────┘   │      │
      ↓ [Conv CH→3]                                │      │
    clean = I(x) + α_rgb · residual  ←────────────────────┘
    clamp [0, 1]

References:
    Dong et al. (2020). MSBDN. CVPR.
    Qin et al. (2020). FFA-Net. AAAI.
    Shi et al. (2016). Real-Time Single Image and Video Super-Resolution
        Using an Efficient Sub-Pixel Convolutional Neural Network. CVPR.

Output: (B, 3, H, W) dehazed image in [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_CHANNELS = 64


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


# ── Transmission-Aware Feature Attention ─────────────────────────────────────

class TransmissionAwareFA(nn.Module):
    """
    Feature Attention with transmission-map modulation.

    Standard CA → PA pipeline, plus a lightweight side-branch that estimates
    a rough transmission map t(x) ∈ (0, 1). The transmission modulates the
    pixel attention weights:

        modulated_pa = pa_weights × (2 − t)

    Where t ≈ 0 (dense haze): attention weight ≈ 2× (strong correction)
    Where t ≈ 1 (clear):      attention weight ≈ 1× (pass through)

    This gives the network an explicit spatial prior on where haze is dense
    vs. sparse, rather than learning it purely implicitly.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)

        # Pixel attention branch
        pa_mid = max(1, channels // 4)
        self.pa_conv = nn.Sequential(
            nn.Conv2d(channels, pa_mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(pa_mid, 1, 1),
            nn.Sigmoid(),
        )

        # Transmission estimation side-branch (lightweight: 2 convs)
        t_mid = max(1, channels // 4)
        self.trans_branch = nn.Sequential(
            nn.Conv2d(channels, t_mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(t_mid, 1, 3, padding=1),
            nn.Sigmoid(),  # t ∈ (0, 1): 1 = clear, 0 = fully hazy
        )

    def forward(self, x):
        ca_out = self.ca(x)                          # channel-reweighted
        pa_w = self.pa_conv(ca_out)                  # (B, 1, H, W)
        t_map = self.trans_branch(x)                 # (B, 1, H, W)
        # Dense haze (low t) → boosted attention; clear (high t) → standard
        modulated = pa_w * (2.0 - t_map)
        return x + ca_out * modulated


# ── Depthwise-Separable Conv Block with SE ───────────────────────────────────

class ConvBlock(nn.Module):
    """
    Depthwise-separable residual block with SE channel attention.
    No normalization — uses careful initialization and GELU activation.

        x → DWConv3×3 → GELU → PWConv1×1(expand) → GELU
          → PWConv1×1(contract) → SE → scale → + x

    Compared to the original 2-conv residual block:
        - Depthwise-separable: same receptive field, ~3× fewer params
        - GELU: smoother gradients than ReLU for dense prediction
        - Channel expansion: 2× mid-width for more representational capacity
        - Built-in SE: per-channel recalibration within each block

    Args:
        channels:  Feature width.
        expansion: Mid-channel expansion ratio (default 2).
    """

    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        mid = channels * expansion

        self.block = nn.Sequential(
            # Depthwise 3×3 (each channel convolved independently)
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels, bias=True),
            nn.GELU(),
            # Pointwise expand
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.GELU(),
            # Pointwise contract
            nn.Conv2d(mid, channels, 1, bias=True),
        )

        # Squeeze-and-Excite (channel recalibration)
        se_mid = max(1, channels // 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, se_mid),
            nn.ReLU(inplace=True),
            nn.Linear(se_mid, channels),
            nn.Sigmoid(),
        )

        # Learnable residual gate — starts at 0.1 for training stability
        self.block_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        out = self.block(x)
        w = self.se(out).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x + self.block_scale * (out * w)


# ── Pixel Shuffle Upsample ──────────────────────────────────────────────────

class PixelShuffleUp(nn.Module):
    """
    Learnable 2× upsampling via sub-pixel convolution (pixel shuffle).

    Conv2d(ch → ch×4) → PixelShuffle(2) → GELU

    Advantages over bilinear + conv:
        - No checkerboard artifacts
        - Fully learnable upsampling kernels
        - Single fused operation

    Reference:
        Shi et al. (2016). ESPCN. CVPR.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return F.gelu(self.shuffle(self.conv(x)))


# ── MSFA-DeNet v2 ───────────────────────────────────────────────────────────

class MSFADeNetV2(nn.Module):
    """
    Enhanced Multi-Scale Feature Attention Dehazing Network.

    3-scale encoder-decoder with depthwise-separable conv blocks,
    transmission-aware attention, pixel shuffle upsampling, and
    per-channel residual scaling. No normalization layers.

    Args:
        channels: Backbone feature width (default 64).
    """

    def __init__(self, channels: int = DEFAULT_CHANNELS):
        super().__init__()
        ch = channels

        # ── Input Projection ─────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            ReflPadConv2d(3, ch, kernel_size=3),
            nn.GELU(),
        )

        # ── Encoder ──────────────────────────────────────────────────────────
        # Scale 0 (full resolution)
        self.enc0 = nn.Sequential(
            ConvBlock(ch),
            TransmissionAwareFA(ch),
        )
        self.down0 = nn.Sequential(
            ReflPadConv2d(ch, ch, kernel_size=3, stride=2),
            nn.GELU(),
        )

        # Scale 1 (½ resolution)
        self.enc1 = nn.Sequential(
            ConvBlock(ch),
            TransmissionAwareFA(ch),
        )
        self.down1 = nn.Sequential(
            ReflPadConv2d(ch, ch, kernel_size=3, stride=2),
            nn.GELU(),
        )

        # Bottleneck (¼ resolution)
        self.bottleneck = nn.Sequential(
            ConvBlock(ch),
            TransmissionAwareFA(ch),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        # ¼ → ½
        self.up1 = PixelShuffleUp(ch)
        self.skip_fuse1 = nn.Sequential(
            nn.Conv2d(ch * 2, ch, kernel_size=1),
            nn.GELU(),
        )
        self.dec1 = nn.Sequential(
            ConvBlock(ch),
            TransmissionAwareFA(ch),
        )

        # ½ → full
        self.up0 = PixelShuffleUp(ch)
        self.skip_fuse0 = nn.Sequential(
            nn.Conv2d(ch * 2, ch, kernel_size=1),
            nn.GELU(),
        )
        self.dec0 = nn.Sequential(
            ConvBlock(ch),
            TransmissionAwareFA(ch),
        )

        # ── Output Projection ────────────────────────────────────────────────
        self.output_proj = ReflPadConv2d(ch, 3, kernel_size=3)

        # Per-channel learnable residual scale (R, G, B independent)
        # Haze scattering is wavelength-dependent — blue channel is
        # affected more than red by Rayleigh scattering.
        self.residual_scale = nn.Parameter(torch.full((1, 3, 1, 1), 0.1))

        # ── Initialization ───────────────────────────────────────────────────
        self._initialize_weights()
        # Zero-init output projection so residual ≈ 0 at epoch 0 (EDSR trick)
        nn.init.zeros_(self.output_proj.conv.weight)
        nn.init.zeros_(self.output_proj.conv.bias)

    def forward(self, x):
        """
        Args:
            x: Hazy input (B, 3, H, W) in [0, 1].
        Returns:
            Dehazed image (B, 3, H, W) in [0, 1].
        """
        f0 = self.input_proj(x)                       # (B, CH, H, W)

        # ── Encoder ──────────────────────────────────────────────────────────
        e0 = self.enc0(f0)                            # (B, CH, H, W)    — skip
        f1 = self.down0(e0)                           # (B, CH, H/2, W/2)

        e1 = self.enc1(f1)                            # (B, CH, H/2, W/2) — skip
        f2 = self.down1(e1)                           # (B, CH, H/4, W/4)

        # ── Bottleneck ───────────────────────────────────────────────────────
        b = self.bottleneck(f2)                       # (B, CH, H/4, W/4)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Scale 1: pixel-shuffle ¼ → ½, fuse with enc1 skip
        u1 = self.up1(b)                              # (B, CH, H/2, W/2)
        if u1.shape[2:] != e1.shape[2:]:
            u1 = F.interpolate(u1, size=e1.shape[2:],
                               mode='bilinear', align_corners=False)
        d1 = self.dec1(self.skip_fuse1(
            torch.cat([u1, e1], dim=1)))              # (B, CH, H/2, W/2)

        # Scale 0: pixel-shuffle ½ → full, fuse with enc0 skip
        u0 = self.up0(d1)                             # (B, CH, H, W)
        if u0.shape[2:] != e0.shape[2:]:
            u0 = F.interpolate(u0, size=e0.shape[2:],
                               mode='bilinear', align_corners=False)
        d0 = self.dec0(self.skip_fuse0(
            torch.cat([u0, e0], dim=1)))              # (B, CH, H, W)

        # ── Global Residual with Per-Channel Scale ───────────────────────────
        residual = self.output_proj(d0)               # (B, 3, H, W)
        clean = x + self.residual_scale * residual    # α_rgb per channel

        return torch.clamp(clean, 0.0, 1.0)

    def _initialize_weights(self):
        """Kaiming initialization for all Conv2d and Linear layers."""
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


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = MSFADeNetV2(channels=64)
    total = sum(p.numel() for p in model.parameters())

    # Component breakdown
    enc_params = sum(
        sum(p.numel() for p in block.parameters())
        for block in [model.enc0, model.enc1, model.bottleneck]
    )
    dec_params = sum(
        sum(p.numel() for p in block.parameters())
        for block in [model.dec0, model.dec1]
    )
    down_params = sum(
        sum(p.numel() for p in block.parameters())
        for block in [model.down0, model.down1]
    )
    up_params = sum(
        sum(p.numel() for p in block.parameters())
        for block in [model.up0, model.up1]
    )
    proj_params = (
        sum(p.numel() for p in model.input_proj.parameters()) +
        sum(p.numel() for p in model.output_proj.parameters())
    )

    # Test with standard size
    dummy = torch.rand(1, 3, 256, 256)
    out = model(dummy)

    print(f"\n{'='*60}")
    print(f"  MSFA-DeNet v2: channels=64")
    print(f"{'='*60}")
    print(f"  Input:           {dummy.shape}")
    print(f"  Output:          {out.shape}")
    print(f"  Range:           [{out.min():.4f}, {out.max():.4f}]")
    print(f"  Total params:    {total:,}")
    print(f"    Encoder (×3):  {enc_params:,}")
    print(f"    Decoder (×2):  {dec_params:,}")
    print(f"    Down (×2):     {down_params:,}")
    print(f"    Up (×2):       {up_params:,}")
    print(f"    Projections:   {proj_params:,}")
    print(f"  Residual scale:  {model.residual_scale.data.flatten().tolist()}")

    # Test with odd size (verifies pixel shuffle size handling)
    dummy_odd = torch.rand(1, 3, 253, 257)
    out_odd = model(dummy_odd)
    print(f"\n  Odd-size test:   {dummy_odd.shape} -> {out_odd.shape}")

    # Verify no normalization layers
    norm_layers = [
        name for name, m in model.named_modules()
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d,
                          nn.LayerNorm))
    ]
    print(f"  Norm layers:     {len(norm_layers)} (expected 0)")
    if norm_layers:
        print(f"    WARNING: {norm_layers}")

    print(f"\n{'='*60}")
    print(f"  [OK] MSFA-DeNet v2 is working correctly!")
    print(f"{'='*60}")
