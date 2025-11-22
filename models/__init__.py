"""
Root package for all model definitions.

Factory helpers live at :mod:`models.factory`, while concrete model families
are organized into subpackages (e.g., :mod:`models.SR` for super-resolution).
"""

from .factory import available_models, build_model_from_config, register_model

__all__ = ["build_model_from_config", "register_model", "available_models"]

