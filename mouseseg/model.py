"""Compact 3D U-Net and ONNX export for mouse organ and tumor CT segmentation."""

import copy
import json
import warnings
from pathlib import Path

import torch
import torch.nn as nn


def fuse_conv_bn(conv, bn):
    """Fold BatchNorm3d statistics into Conv3d weights and bias for fast inference."""
    fused = nn.Conv3d(
        conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=True,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)
    bias = conv.bias.detach() if conv.bias is not None else torch.zeros_like(bn.running_mean)
    scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
    fused.weight.data.copy_(conv.weight.detach() * scale.view(-1, 1, 1, 1, 1))
    fused.bias.data.copy_((bias - bn.running_mean.detach()) * scale + bn.bias.detach())
    return fused.eval()


class ConvBNAct3d(nn.Module):
    """Basic 3D convolution + BatchNorm3d + optional ReLU block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, act=True):
        super().__init__()
        pad = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, groups=groups, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuse_in_place(self):
        if isinstance(self.bn, nn.BatchNorm3d):
            self.conv = fuse_conv_bn(self.conv, self.bn)
            self.bn = nn.Identity()


class DepthwiseSeparableResBlock3d(nn.Module):
    """Residual block using two 3x3x3 depthwise + 1x1x1 pointwise 3D convolutions."""

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.dw1 = ConvBNAct3d(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels)
        self.pw1 = ConvBNAct3d(in_channels, out_channels, kernel_size=1, padding=0)
        self.drop = nn.Dropout3d(p=dropout) if dropout > 0.0 else nn.Identity()
        self.dw2 = ConvBNAct3d(out_channels, out_channels, kernel_size=3, groups=out_channels)
        self.pw2 = ConvBNAct3d(out_channels, out_channels, kernel_size=1, padding=0, act=False)
        self.shortcut = (
            ConvBNAct3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, act=False)
            if (stride != 1 or in_channels != out_channels) else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.pw2(self.dw2(self.drop(self.pw1(self.dw1(x)))))
        return self.act(out + self.shortcut(x))


class LargeKernelContextBlock3d(nn.Module):
    """5x5x5 depthwise residual context block at the 1/16-scale bottleneck."""

    def __init__(self, channels):
        super().__init__()
        self.dw = ConvBNAct3d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.pw = ConvBNAct3d(channels, channels, kernel_size=1, padding=0, act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.pw(self.dw(x)) + x)


def make_stage3d(in_channels, out_channels, num_blocks, stride, dropout=0.0):
    layers = [DepthwiseSeparableResBlock3d(in_channels, out_channels, stride=stride, dropout=dropout)]
    for _ in range(num_blocks - 1):
        layers.append(DepthwiseSeparableResBlock3d(out_channels, out_channels, stride=1, dropout=dropout))
    return nn.Sequential(*layers)


class UpStage3d(nn.Module):
    """Decoder stage: 1x1x1 projection + 2x trilinear upsampling + skip concatenation."""

    def __init__(self, in_channels, skip_channels, out_channels, num_blocks=1):
        super().__init__()
        self.up = nn.Sequential(
            ConvBNAct3d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=2.0, mode="trilinear", align_corners=False),
        )
        self.fuse_skip = make_stage3d(out_channels + skip_channels, out_channels, num_blocks, stride=1)

    def forward(self, x, skip):
        return self.fuse_skip(torch.cat([self.up(x), skip], dim=1))


class CompactUNet3D(nn.Module):
    """Compact 3D U-Net mapping [B, C, D, H, W] CT patches to (organ_logits, tumor_logits)."""

    def __init__(self, in_channels=2, num_organ_classes=12, base_channels=32, channel_multipliers=(1, 2, 4, 6),
                 blocks_per_stage=2, decoder_blocks_per_stage=1, edge_skip_channels=12, enable_organ_head=True, dropout=0.0):
        super().__init__()
        channels = [base_channels * m for m in channel_multipliers]
        num_stages = len(channels)
        self.in_channels = in_channels
        self.enable_organ_head = enable_organ_head
        self.tumor_only = not enable_organ_head
        self.num_organ_classes = 1 if self.tumor_only else num_organ_classes
        self.spatial_divisor = 2 ** num_stages

        self.edge_skip = ConvBNAct3d(in_channels, edge_skip_channels, kernel_size=3, stride=1)
        self.stem = ConvBNAct3d(in_channels, channels[0], kernel_size=3, stride=2)
        self.enc0 = make_stage3d(channels[0], channels[0], blocks_per_stage, stride=1)

        self.encoders = nn.ModuleList()
        for i in range(1, num_stages):
            drop = dropout if i == num_stages - 1 else 0.0
            stage = make_stage3d(channels[i - 1], channels[i], blocks_per_stage, stride=2, dropout=drop)
            if i == num_stages - 1:
                stage.append(LargeKernelContextBlock3d(channels[i]))
            self.encoders.append(stage)

        self.decoders = nn.ModuleList(
            UpStage3d(channels[i], channels[i - 1], channels[i - 1], decoder_blocks_per_stage)
            for i in range(num_stages - 1, 0, -1)
        )
        self.up_to_full = nn.Sequential(
            ConvBNAct3d(channels[0], edge_skip_channels, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=2.0, mode="trilinear", align_corners=False),
        )
        self.full_refine = ConvBNAct3d(edge_skip_channels, edge_skip_channels, kernel_size=3, stride=1)

        # Keep attribute names oar_head / tumor_head so pretrained .pt weights load 1-for-1
        self.oar_head = nn.Conv3d(edge_skip_channels, self.num_organ_classes, kernel_size=1)
        if self.tumor_only:
            nn.init.zeros_(self.oar_head.weight)
            nn.init.zeros_(self.oar_head.bias)
            self.oar_head.requires_grad_(False)
        self.tumor_head = nn.Conv3d(edge_skip_channels, 1, kernel_size=1)

    def forward(self, x):
        edge_features = self.edge_skip(x)
        features = self.enc0(self.stem(x))
        skips = [features]
        for encoder in self.encoders:
            features = encoder(features)
            skips.append(features)
        decoded = skips[-1]
        for i, up_stage in enumerate(self.decoders):
            decoded = up_stage(decoded, skips[-(i + 2)])
        full_features = self.full_refine(self.up_to_full(decoded) + edge_features)
        return self.oar_head(full_features), self.tumor_head(full_features)

    def fuse_for_inference(self):
        fused_model = copy.deepcopy(self).eval()
        for module in fused_model.modules():
            if isinstance(module, ConvBNAct3d):
                module.fuse_in_place()
        return fused_model


def build_model(mode="unified", dropout=0.0):
    """Create the MouseSeg 3D U-Net for 'unified' (10 organs + tumor) or 'tumor_only'."""
    if mode == "unified":
        return CompactUNet3D(in_channels=2, num_organ_classes=12, edge_skip_channels=12, enable_organ_head=True, dropout=dropout)
    if mode in ("tumor_only", "specialist"):
        return CompactUNet3D(in_channels=1, num_organ_classes=1, edge_skip_channels=8, enable_organ_head=False, dropout=dropout)
    raise ValueError(f"Unknown mode '{mode}'. Choose 'unified' or 'tumor_only'.")


def export_onnx(model, output_path, mode):
    """Fold BatchNorm layers, export dynamic-axis ONNX, and embed MouseSeg inference metadata."""
    import onnx

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fused = model.cpu().fuse_for_inference().eval()
    dummy = torch.zeros(1, fused.in_channels, 64, 96, 96, dtype=torch.float32)
    axes = {k: {0: "batch", 2: "depth", 3: "height", 4: "width"} for k in ("input", "oar_logits", "tumor_logits")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            fused, dummy, str(output_path), opset_version=17,
            input_names=["input"], output_names=["oar_logits", "tumor_logits"],
            dynamic_axes=axes, dynamo=False,
        )

    unified = mode == "unified"
    meta = {
        "arch": "compact_unet3d", "is_3d": True, "in_channels": 2 if unified else 1, "spatial_divisor": 16,
        "num_oar_classes": 12 if unified else 1, "tumor_only": not unified, "tumor_label_id": 12,
        "target_spacing_mm": [0.2, 0.2, 0.2], "hu_min": -1000.0 if unified else -400.0, "hu_max": 3000.0 if unified else 1000.0,
        "hu_windows": [[-1000.0, 3000.0], [-400.0, 1000.0]] if unified else [[-400.0, 1000.0]],
        "harmonize_air_floor": True, "soft_tissue_calib_target_hu": 45.0, "soft_tissue_max_shift_hu": 25.0,
        "protect_internal_oars": True, "recommended_disable_thigh_muscle": True,
        "recommended_max_3d_depth_voxels": 128, "window_slices": 128, "window_overlap": 0.25,
        "recommended_tumor_threshold": 0.5, "recommended_tumor_seed_threshold": 0.78 if unified else 0.80,
        "recommended_tumor_grow_threshold": 0.18 if unified else 0.40, "recommended_tumor_min_seed_voxels": 4,
        "recommended_min_tumor_vol_mm3": 6.0, "recommended_rel_tumor_cc_ratio": 0.1, "tumor_smooth_sigma": 0.5,
    }
    if unified:
        meta.update({
            "recommended_oar_min_prob": {"2": 0.6, "5": 0.6, "9": 0.35},
            "recommended_bladder_prob_threshold": 0.18,
            "recommended_stomach_seed_threshold": 0.5,
            "recommended_stomach_grow_threshold": 0.15,
        })

    onnx_model = onnx.load(str(output_path))
    meta_str = json.dumps(meta, sort_keys=True)
    for key in ("mouseseg_config", "mouseseg_config_json"):
        entry = onnx_model.metadata_props.add()
        entry.key, entry.value = key, meta_str
    onnx.save(onnx_model, str(output_path))
