# Registering custom models in JHTDB-ML

The repository exposes a small factory (`models/factory.py`) that converts YAML configuration entries into actual `torch.nn.Module` instances. Registering a new model is a two-step process:

1. Implement the module in `models/`.
2. Decorate a builder function with `@register_model("name")` so the factory can construct it.

```python
# models/my_model.py
import torch.nn as nn

class MySRModel(nn.Module):
    def __init__(self, scale: int, hidden_dim: int = 64):
        super().__init__()
        # define layers ...

    def forward(self, x):
        return x
```

```python
# models/factory.py
@register_model("my_model")
def _build_my_model(cfg: Dict, patch_size_lr: int, patch_size_hr: int) -> nn.Module:
    params = cfg.get("my_model", {})
    scale = int(cfg.get("scale_factor", patch_size_hr // patch_size_lr))
    return MySRModel(scale=scale, hidden_dim=int(params.get("hidden_dim", 64)))
```

Once registered, you can point any experiment YAML at the new architecture:

```yaml
model:
  architecture: "my_model"
  scale_factor: 4
  my_model:
    hidden_dim: 128
```

The notebook pipeline will automatically pick it up without additional code changes.
