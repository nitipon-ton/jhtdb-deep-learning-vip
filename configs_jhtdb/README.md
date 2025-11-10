# JHTDB Data Fetching Configurations

This directory contains YAML configuration files for fetching different types of flow data from the Johns Hopkins Turbulence Database (JHTDB).

**🚀 New user?** See [QUICK_START.md](QUICK_START.md) for a fast introduction!

---

## Available Configurations

### 1. Channel Flow Configurations

#### `channel_xz_wall_parallel.yaml`
- **Plane**: Wall-parallel (x-z) at y = 0.9
- **Use case**: Near-wall turbulence, streaks, vortices
- **Domain**: 2π × π (streamwise × spanwise)
- **Resolution**: 512×512 (matches DNS)
- **Output**: `data/raw/channel_sr_512/`

#### `channel_xy_wall_normal.yaml`
- **Plane**: Wall-normal (x-y) at z = 1.5π
- **Use case**: Cross-stream structures, wall-normal velocity
- **Domain**: 2π × 2 (streamwise × wall-normal, y ∈ [-1, 1])
- **Resolution**: 512×512 (matches DNS)
- **Output**: `data/raw/channel_xy_512/`

#### `channel_yz_wall_normal.yaml`
- **Plane**: Wall-normal (y-z) at x = π
- **Use case**: Spanwise-wall-normal structures
- **Domain**: 2 × π (wall-normal × spanwise, y ∈ [-1, 1])
- **Resolution**: 512×512 (matches DNS)
- **Output**: `data/raw/channel_yz_512/`

### 2. Isotropic Turbulence Configuration

#### `isotropic_xy_plane.yaml`
- **Plane**: Arbitrary x-y plane at z = π
- **Use case**: Fundamental turbulence, isotropy studies
- **Domain**: π × π (homogeneous, any plane equivalent)
- **Resolution**: 512×512 (matches DNS)
- **Dataset**: `isotropic1024coarse` (Re_λ ≈ 433)
- **Output**: `data/raw/isotropic_xy_512/`

---

## Configuration Structure

Each configuration file contains:

```yaml
experiment:          # Experiment metadata
  name: ...
  description: ...

dataset:             # JHTDB dataset information
  name: ...          # "channel" or "isotropic1024coarse"
  variable: ...      # "velocity", "pressure", etc.

plane:               # Plane specification
  type: ...          # "wall_parallel", "wall_normal", "arbitrary"
  orientation: ...   # "xz", "xy", "yz"
  fixed_coordinate:  # Which coordinate is fixed
    name: ...        # "x", "y", or "z"
    value: ...       # Position

domain:              # Spatial extent
  [coord1]:
    min: ...
    max: ...
  [coord2]:
    min: ...
    max: ...
  grid_size: 512     # Resolution

temporal:            # Time configuration
  num_snapshots: 500
  time_start: 0.0
  time_end: 13.0

output:              # Where to save data
  base_dir: "data/raw"
  experiment_dir: "..."
  h5_filename: "..."

# ... more settings
```

---

## How to Use

### In Jupyter Notebook

```python
import yaml
from pathlib import Path

# Load configuration
config_path = Path("../configs_jhtdb/channel_xz_wall_parallel.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Access parameters
DATASET = config['dataset']['name']
GRID_SIZE = config['domain']['grid_size']
X_MIN = config['domain']['x']['min']
X_MAX = config['domain']['x']['max']
# ... etc
```

### Switching Configurations

To fetch different data, simply load a different config file:

```python
# Wall-parallel channel flow
config_path = Path("../configs_jhtdb/channel_xz_wall_parallel.yaml")

# Wall-normal channel flow
config_path = Path("../configs_jhtdb/channel_xy_wall_normal.yaml")

# Isotropic turbulence
config_path = Path("../configs_jhtdb/isotropic_xy_plane.yaml")
```

---

## Resolution Matching

All configurations are designed to **match DNS resolution** by:

1. **Full domain**: Would exceed 512×512 with DNS spacing
2. **Reduced domain**: Use fraction of domain to fit 512×512
3. **Result**: Exact DNS resolution, no interpolation losses

### Examples:

**Channel Flow (full domain: 8π × 3π, grid: 2048 × 1536)**
- DNS spacing: dx = 8π/2048 ≈ 0.0123
- Our domain: 2π × π (1/4 × 1/3 of full)
- Our grid: 512 × 512
- Our spacing: 2π/512 ≈ 0.0123 ✓ **MATCHES!**

**Isotropic (full domain: 2π × 2π, grid: 1024 × 1024)**
- DNS spacing: dx = 2π/1024 ≈ 0.0061
- Our domain: π × π (1/2 × 1/2 of full)
- Our grid: 512 × 512
- Our spacing: π/512 ≈ 0.0061 ✓ **MATCHES!**

---

## Customization

### Modify Existing Config

To customize, edit the YAML file directly:

```yaml
temporal:
  num_snapshots: 1000    # Increase snapshots
  time_start: 0.0
  time_end: 25.0         # Extend time range

domain:
  grid_size: 256         # Reduce resolution (if token limited)
```

### Create New Config

Copy an existing config and modify:

```bash
cp channel_xz_wall_parallel.yaml my_custom_config.yaml
# Edit my_custom_config.yaml
```

---

## Dataset Information

### Channel Flow (`channel`)
- **Domain**: 8π × 2 × 3π (streamwise × wall-normal × spanwise)
- **Grid**: 2048 × 512 × 1536
- **Re_τ**: 1000 (friction Reynolds number)
- **Boundaries**: Periodic (x, z), No-slip walls (y)
- **Time range**: 0 to 26 s

### Isotropic Turbulence (`isotropic1024coarse`)
- **Domain**: 2π × 2π × 2π (cubic)
- **Grid**: 1024³
- **Re_λ**: 433 (Taylor microscale Reynolds number)
- **Boundaries**: Periodic (all directions)
- **Time range**: 0 to 10.056 s (5028 timesteps, dt=0.0002)
- **Forcing**: Large-scale forcing for statistical stationarity

---

## Plane Orientations

### Wall-Parallel (x-z)
```
      z (spanwise)
      ↑
      |
      |_____→ x (streamwise)

Fixed: y (wall-normal)
```

### Wall-Normal (x-y)
```
      y (wall-normal)
      ↑
      |
      |_____→ x (streamwise)

Fixed: z (spanwise)
```

### Wall-Normal (y-z)
```
      z (spanwise)
      ↑
      |
      |_____→ y (wall-normal)

Fixed: x (streamwise)
```

---

## Token Requirements

### High-Capacity Token (Recommended)
- **Limit**: 500,000 points per query
- **512×512**: ✓ (262,144 points = 52% utilization)
- **Config files designed for this token**

### Standard Token
- **Limit**: 4,096 points per query
- **Requires tiling**: Use alternative section in notebook
- **256×256**: 16 tiles needed

---

## Output Structure

Each configuration saves data to a unique directory:

```
data/raw/
├── channel_sr_512/           # Wall-parallel (x-z)
│   ├── channel_x-z_y0.9_t*.npz
│   ├── visualizations/
│   └── channel_sr_512.h5
├── channel_xy_512/           # Wall-normal (x-y)
│   ├── channel_x-y_z*.npz
│   ├── visualizations/
│   └── channel_xy_512.h5
├── channel_yz_512/           # Wall-normal (y-z)
│   ├── channel_y-z_x*.npz
│   ├── visualizations/
│   └── channel_yz_512.h5
└── isotropic_xy_512/         # Isotropic (x-y)
    ├── isotropic_x-y_z*.npz
    ├── visualizations/
    └── isotropic_xy_512.h5
```

---

## Best Practices

1. **Match DNS resolution**: Use provided domain limits
2. **Separate experiments**: Each config has unique output directory
3. **Document changes**: Update `notes` field in YAML
4. **Version control**: Keep configs in git (not data)
5. **Share configs**: Easy to reproduce experiments

---

## Troubleshooting

### "File not found" error
→ Check `config_path` points to correct `.yaml` file

### "Token limit exceeded"
→ Reduce `grid_size` or use tiling approach

### "Domain out of bounds"
→ Check dataset limits (8π for channel, 2π for isotropic)

### "Wrong plane orientation"
→ Verify `fixed_coordinate.name` matches your desired plane

---

## Future Extensions

Possible additions:
- Higher Reynolds number channel (`channel5200`)
- Finer isotropic turbulence (`isotropic1024fine`)
- Different time ranges
- 3D subvolumes (not just planes)
- Multiple planes simultaneously

---

**Created**: 2025-11-04
**Version**: 1.0
**Maintainer**: Aditya Sai Pranith Ayapilla
