"""
Convenience imports for the model zoo.

The individual model files contain the actual implementations. The factory
module exposes a lightweight registry so experiments can refer to architectures
by name in YAML configuration files.

Available architectures:
- unet: Baseline U-Net for super-resolution
- hdsc_ms: Hybrid DSC/MS (Fukami et al. 2019)
- edsr: Enhanced Deep SR (Lim et al. 2017) - proven SOTA
- mlp: MLP baseline for super-resolution

See models/factory.py for registration details.
"""

from models.factory import available_models, build_model_from_config, register_model
from models.hdsc_ms import HybridDSCMSR, HDSCMS
from models.mlp_sr import MLPSuperResolution, TinyMLP
from models.unet_sr import UNetSRx4
from models.edsr import EDSR, EDSRLite

__all__ = [
    # Factory functions
    "build_model_from_config",
    "register_model",
    "available_models",
    # SR Models
    "UNetSRx4",
    "HybridDSCMSR",
    "HDSCMS",
    "EDSR",
    "EDSRLite",
    "MLPSuperResolution",
    "TinyMLP",
]
