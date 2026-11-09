"""
Model factory and registry utilities.

Provides a central place to construct models from configuration without
scattering architecture-specific logic throughout notebooks or scripts.

Registered models:
- unet: Baseline U-Net for SR
- hdsc_ms: Hybrid DSC/MS (Fukami et al. 2019 inspired)
- edsr: Enhanced Deep SR (Lim et al. 2017) - proven SOTA architecture
- mlp: Fully-connected MLP baseline
- residual_cnn: Full-resolution periodic CNN that predicts a residual correction
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch.nn as nn

from models.SR.mlp_sr import MLPSuperResolution, TinyMLP
from models.SR.unet_sr import UNetSRx4, UNetSRx8
from models.SR.hdsc_ms import HybridDSCMSR
from models.SR.edsr import EDSR, EDSRLite
from models.SR.unet_meteo import MeteorologyUNetSR
from models.SR.esrgan import ESRGANGenerator
from models.SR.swinir import SwinIRLite
from models.SR.residual_cnn import ResidualCorrectionCNN2d

ModelBuilder = Callable[[Dict, int, int], nn.Module]
_MODEL_REGISTRY: Dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered.")
        _MODEL_REGISTRY[name] = builder
        return builder

    return decorator


def build_model_from_config(
    model_cfg: Dict,
    patch_size_lr: int,
    patch_size_hr: int,
) -> nn.Module:
    """
    Build a model from configuration dictionary.

    Args:
        model_cfg: Model configuration with 'architecture' key
        patch_size_lr: LR patch size (e.g., 64)
        patch_size_hr: HR patch size (e.g., 256)

    Returns:
        Instantiated model
    """
    arch = model_cfg.get("architecture")
    if not arch:
        raise KeyError("Model config missing 'architecture' key.")

    try:
        builder = _MODEL_REGISTRY[arch]
    except KeyError as exc:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown architecture '{arch}'. Available: {available or 'none'}"
        ) from exc

    return builder(model_cfg, patch_size_lr, patch_size_hr)


def available_models() -> Tuple[str, ...]:
    """Return tuple of registered model names."""
    return tuple(sorted(_MODEL_REGISTRY.keys()))


# ============================================================================
# Model Builders
# ============================================================================

@register_model("unet")
def _build_unet(model_cfg: Dict, _: int, __: int) -> nn.Module:
    """Build baseline U-Net for SR (auto-selects x4 or x8 based on scale)."""
    params = model_cfg.get("unet", {})
    in_ch = int(params.get("in_ch", 1))
    out_ch = int(params.get("out_ch", 1))
    base_ch = int(params.get("base_ch", 32))
    scale = int(model_cfg.get("scale_factor", 4))

    # Auto-select architecture based on scale
    if scale == 8:
        return UNetSRx8(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch, scale=scale)
    elif scale == 4:
        return UNetSRx4(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch, scale=scale)
    else:
        raise ValueError(f"U-Net only supports scale=4 or scale=8, got scale={scale}")


@register_model("hdsc_ms")
def _build_hdsc_ms(model_cfg: Dict, patch_size_lr: int, patch_size_hr: int) -> nn.Module:
    """Build hybrid DSC/MS U-Net for turbulence SR."""
    params = model_cfg.get("hdsc_ms", {})
    in_ch = int(params.get("in_ch", 1))
    out_ch = int(params.get("out_ch", 1))
    scale = int(model_cfg.get("scale_factor", patch_size_hr // patch_size_lr))
    return HybridDSCMSR(
        in_ch=in_ch,
        out_ch=out_ch,
        scale=scale,
        lr_size=patch_size_lr,
        hr_size=patch_size_hr,
        dsc_width=int(params.get("dsc_width", 32)),
        ms_mid=int(params.get("ms_mid", 16)),
        ms_small=int(params.get("ms_small", 8)),
        ms_out=int(params.get("ms_out", 3)),
        upsample_mode=params.get("upsample_mode", "bilinear"),
    )


@register_model("unet_meteo")
def _build_unet_meteo(model_cfg: Dict, patch_size_lr: int, patch_size_hr: int) -> nn.Module:
    params = model_cfg.get("unet_meteo", {})
    feature_channels = params.get("feature_channels", [32, 64, 128, 256])
    return MeteorologyUNetSR(
        in_channels=int(params.get("in_ch", 1)),
        out_channels=int(params.get("out_ch", 1)),
        scale_factor=int(model_cfg.get("scale_factor", patch_size_hr // patch_size_lr)),
        feature_channels=feature_channels,
        n_blocks_per_stage=int(params.get("n_blocks_per_stage", 2)),
        num_latent_layers=int(params.get("num_latent_layers", 2)),
        pre_upsample_mode=params.get("pre_upsample_mode", "bilinear"),
    )


@register_model("mlp")
def _build_mlp(model_cfg: Dict, patch_size_lr: int, patch_size_hr: int) -> nn.Module:
    """Build MLP baseline for SR."""
    params = model_cfg.get("mlp", {})
    variant = params.get("variant", "standard")  # "standard" or "tiny"

    if variant == "tiny":
        return TinyMLP(
            lr_size=patch_size_lr,
            hr_size=patch_size_hr,
            hidden_dim=int(params.get("hidden_dim", 512)),
            num_layers=int(params.get("num_layers", 3)),
        )
    else:
        return MLPSuperResolution(
            lr_size=patch_size_lr,
            hr_size=patch_size_hr,
            hidden_dims=params.get("hidden_dims", [1024, 2048, 4096, 2048, 1024]),
            dropout_p=float(params.get("dropout_p", 0.1)),
        )


@register_model("edsr")
def _build_edsr(model_cfg: Dict, _: int, __: int) -> nn.Module:
    """Build EDSR (Enhanced Deep SR) network."""
    params = model_cfg.get("edsr", {})
    scale = int(model_cfg.get("scale_factor", 4))

    # Check if using lite variant
    variant = params.get("variant", "standard")  # "standard" or "lite"

    if variant == "lite":
        return EDSRLite(
            scale=scale,
            n_resblocks=int(params.get("n_resblocks", 8)),
            n_feats=int(params.get("n_feats", 32)),
            in_channels=int(params.get("in_channels", 1)),
            out_channels=int(params.get("out_channels", 1)),
        )
    else:
        return EDSR(
            scale=scale,
            n_resblocks=int(params.get("n_resblocks", 16)),
            n_feats=int(params.get("n_feats", 64)),
            res_scale=float(params.get("res_scale", 1.0)),
            in_channels=int(params.get("in_channels", 1)),
            out_channels=int(params.get("out_channels", 1)),
        )
@register_model("esrgan")
def _build_esrgan(model_cfg: Dict, patch_size_lr: int, patch_size_hr: int) -> nn.Module:
    """Build ESRGAN-style RRDB generator."""
    params = model_cfg.get("esrgan", {})
    scale = int(model_cfg.get("scale_factor", patch_size_hr // patch_size_lr))
    return ESRGANGenerator(
        in_channels=int(params.get("in_ch", 1)),
        out_channels=int(params.get("out_ch", 1)),
        nf=int(params.get("nf", 64)),
        gc=int(params.get("gc", 32)),
        num_blocks=int(params.get("num_blocks", 23)),
        scale=scale,
    )


@register_model("swinir")
def _build_swinir(model_cfg: Dict, patch_size_lr: int, patch_size_hr: int) -> nn.Module:
    """Build a lightweight SwinIR generator."""
    params = model_cfg.get("swinir", {})
    scale = int(model_cfg.get("scale_factor", patch_size_hr // patch_size_lr))
    return SwinIRLite(
        in_ch=int(params.get("in_ch", 1)),
        out_ch=int(params.get("out_ch", 1)),
        embed_dim=int(params.get("embed_dim", 64)),
        depths=int(params.get("depths", 6)),
        num_heads=int(params.get("num_heads", 4)),
        window_size=int(params.get("window_size", 8)),
        mlp_ratio=float(params.get("mlp_ratio", 2.0)),
        scale=scale,
    )


@register_model("residual_cnn")
def _build_residual_cnn(
    model_cfg: Dict,
    patch_size_lr: int,
    patch_size_hr: int,
) -> nn.Module:
    """Build the full-resolution residual correction CNN."""
    params = model_cfg.get("residual_cnn", {})
    scale = int(model_cfg.get("scale_factor", patch_size_hr // patch_size_lr))
    if patch_size_lr * scale != patch_size_hr:
        raise ValueError(
            f"Residual CNN scale {scale} is incompatible with "
            f"LR/HR sizes {patch_size_lr}/{patch_size_hr}."
        )
    return ResidualCorrectionCNN2d(
        scale=scale,
        in_channels=int(params.get("in_ch", 1)),
        out_channels=int(params.get("out_ch", 1)),
        hidden_channels=int(params.get("hidden_channels", 32)),
        num_residual_blocks=int(params.get("num_residual_blocks", 4)),
        kernel_size=int(params.get("kernel_size", 3)),
        dilation=int(params.get("dilation", 1)),
        normalization=params.get("normalization", "batch"),
        group_norm_groups=int(params.get("group_norm_groups", 8)),
        activation=params.get("activation", "relu"),
        dropout_rate=float(params.get("dropout_rate", 0.0)),
        upsample_mode=params.get("upsample_mode", "bicubic"),
    )
