"""
Super-resolution model subpackage.

This namespace groups concrete SR architectures; the registry in
``models.factory`` is responsible for instantiation.
"""

from .edsr import EDSR, EDSRLite
from .hdsc_ms import HybridDSCMSR
from .mlp_sr import MLPSuperResolution, TinyMLP
from .unet_sr import UNetSRx4, UNetSRx8
from .unet_meteo import MeteorologyUNetSR
from .unet_meteo_3d import MeteorologyUNetSR3D
from .esrgan import ESRGANGenerator, ESRGANDiscriminator
from .swinir import SwinIRLite

__all__ = [
    "EDSR",
    "EDSRLite",
    "HybridDSCMSR",
    "MLPSuperResolution",
    "TinyMLP",
    "UNetSRx4",
    "UNetSRx8",
    "MeteorologyUNetSR",
    "MeteorologyUNetSR3D",
    "ESRGANGenerator",
    "ESRGANDiscriminator",
    "SwinIRLite",
]
