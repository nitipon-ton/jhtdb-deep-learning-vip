# Standalone 3D Hybrid Fourier Neural Operator (FNO) for JHTDB velocity super-resolution

**This module lives at `models/SR/fno_3d/` inside `nitipon-ton/jhtdb-deep-learning-vip`. It is self-contained and does not modify the existing 3D U-Net, model factory, training pipeline, or saved checkpoints.**

The goal is to compare a hybrid 3D Fourier Neural Operator with the repo's **3D Meteorology U-Net**, using **the same existing three-component velocity cube dataset and residual-correction targets**. This is a proposed research model, **not a claim that FNO will outperform U-Net**.

## 1. What the model does (beginner explanation)

- Input: an interpolated, initially blurry **3D velocity cube** (u, v, w), already expanded to its final resolution (commonly 128×128×128).
- Fourier layers: learn correlations between different spatial frequencies across the whole 3D cube.
- Local convolutions: learn nearby structures, such as local gradients and swirling patterns.
- Residual blocks: help learning inside the network by preserving previous feature information.
- Output: **three correction fields** that should be added to the input to approximate the true DNS velocity field.

**Important:** This is a *3D* FNO, not the earlier illustrative 2D version. The repository's 3D U-Net consumes volumetric velocity data. A 2D FNO would not be an apples-to-apples alternative for those 3D cube inputs.

```text
input  = blurred/downsampled/upsampled velocity, shape (B,3,128,128,128)
target = DNS - input,                       shape (B,3,128,128,128)
model(input) = predicted correction,        shape (B,3,128,128,128)
predicted_DNS = input + model(input)
```

This **matches** `configs/sr_unet_meteo_3d_velocity128.yaml` in the inspected GitHub repo: `scale_factor: 1`, `residual: false` (meaning the existing 3D U-Net also predicts the correction). **Do not add a second global residual to the model output during training:** the target is already a correction. For evaluation or inference, add the predicted correction to the input once.

The 3D FNO does **not** start with bilinear upsampling: the repo's paired 3D `.npy` input arrays are *already* upsampled to the full cube shape. It instead performs Fourier-domain mixing directly on that grid.

## 2. Contents

```text
models/SR/fno_3d/
  __init__.py          # importable package (HybridFNO3D, SpectralConv3d)
  model.py             # 3D Fourier spectral conv + local conv + residual FNO blocks
  dataset.py           # existing repo-compatible 3D NPY residual pairs
  losses.py            # MSE + 3D gradient + optional spectral / divergence losses
  train.py             # train, evaluate, save checkpoints and metrics
  config.yaml          # all defaults you can change
  sweep.py             # optional sequential validation-only hyperparameter trials
  make_synthetic_data.py # tiny FAKE cubes to check wiring, not physics
  requirements.txt
  tests/test_fno.py    # unit tests
  README.md
```

## 3. Setup (without touching the U-Net)

Clone the existing repository (if you have not already), then run these commands **from `models/SR/fno_3d/`**. Existing U-Net files are not changed. The FNO is intentionally not registered with `models/factory.py`; use its own `train.py` entry point.

Python 3.11 recommended. Install PyTorch using the appropriate instructions for your actual GPU/cluster first (consult [pytorch.org](https://pytorch.org/get-started/locally/)), then install the remaining dependencies:

```bash
git clone https://github.com/nitipon-ton/jhtdb-deep-learning-vip.git
cd jhtdb-deep-learning-vip/models/SR/fno_3d
python -m venv .venv
source .venv/bin/activate                   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import torch; print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

On a PACE cluster, run GPU training **inside a GPU allocation** with a CUDA-capable PyTorch environment. The scripts run on CPU too, but full 128³ 3D training on CPU is impractical.

## 4. Point FNO to the **existing** processed velocity cubes

The 3D U-Net config uses:

```yaml
data:
  processed_dir: ../superresolution-nn/data/processed/cnn
```

That location is **not necessarily the right absolute path on your machine**. Find the *actual* directory that contains the three split folders:

```text
/absolute/path/to/superresolution-nn/data/processed/cnn/
  train/
    input_t0001.npy
    target_t0001.npy
    ...
  val/
    input_t....npy
    target_t....npy
  test/
    input_t....npy
    target_t....npy
```

Each `.npy` is `(Z,Y,X,3)`, where the last axis stores u, v, w. The **input** is already coarse-to-fine interpolated; the **target** is the residual `DNS - input`. The loader converts it to channels-first `(3,Z,Y,X)` for PyTorch.

**Do not modify these files.** Use a filesystem path to read them directly. When launching the FNO, override `data.processed_dir` with that directory (no spaces in paths is simplest):

```bash
python train.py \
  -o data.processed_dir=/absolute/path/to/superresolution-nn/data/processed/cnn \
  -o experiment.run_name=fno3d_first_run
```

If the 3D input/target pairs have not been generated yet, follow the *existing* `superresolution-nn` preprocessing procedure first. This project doesn't fetch or synthesize real DNS data.

**Why is there no `scale_factor: 4` in the FNO model?** The saved input is already at full resolution. Re-upsampling would create an incorrectly sized prediction.

## 5. First run: small synthetic smoke test

This quick test checks whether the **code wiring works**, not whether it learns real turbulence. Synthetic cubes are tiny (16³), so the spectral modes must be small.

```bash
python make_synthetic_data.py --output synthetic_data --size 16
python train.py \
  -o data.processed_dir=synthetic_data \
  -o experiment.run_name=fno3d_smoke \
  -o model.width=4 \
  -o model.modes='[3,3,3]' \
  -o model.depth=1 \
  -o training.epochs=2 \
  -o training.batch_size=2 \
  -o hardware.device=cpu
python train.py --evaluate --split test \
  -o data.processed_dir=synthetic_data \
  -o experiment.run_name=fno3d_smoke \
  -o hardware.device=cpu
```

A synthetic dataset was used to test this package's pipeline. Your hardware, model configuration and real data can behave differently.

## 6. Train on real 3D JHTDB velocity cubes

Use the real processed directory and a **new run name**:

```bash
python train.py \
  -o data.processed_dir=/absolute/path/to/processed/cnn \
  -o experiment.run_name=fno3d_baseline \
  -o hardware.device=cuda
```

Initial config is deliberately conservative for volumetric data:

```yaml
model:
  width: 16
  modes: [8, 8, 8]
  depth: 3
  local_kernel: 3
  local_path: true
  residual_blocks: true
  coordinates: false
training:
  epochs: 100
  batch_size: 1
  lr: 0.0005
  amp: false
loss:
  pixel_weight: 1.0
  gradient_weight: 0.1
  spectral_weight: 0.0
  divergence_weight: 0.0
```

A run creates a **separate** directory inside `models/SR/fno_3d/`:

```text
runs/fno3d_baseline/
  config_used.yaml
  history.csv
  checkpoints/
    best_model.pth    # lowest validation objective
    last_model.pth    # most recent epoch, for resuming
```

`history.csv` logs losses and learning rates. The best checkpoint is chosen **only from the validation split**, not the test split. Training stops early if validation does not improve for 20 epochs. New runs refuse to overwrite existing checkpoints; change `experiment.run_name` for each trial.

If training is interrupted, resume using the original config/model settings:

```bash
python train.py \
  -o data.processed_dir=/absolute/path/to/processed/cnn \
  -o experiment.run_name=fno3d_baseline \
  --resume runs/fno3d_baseline/checkpoints/last_model.pth
```

You may increase `training.epochs` when resuming. For clean comparisons, avoid changing learning rate, scheduler or loss weights partway through a run. Only load checkpoints you trust.

## 7. Evaluate (after choosing hyperparameters)

After selecting your preferred FNO model based on **validation only**, evaluate its best checkpoint on the held-out test data:

```bash
python train.py --evaluate --split test --save-predictions \
  -o data.processed_dir=/absolute/path/to/processed/cnn \
  -o experiment.run_name=fno3d_baseline \
  -o hardware.device=cuda
```

Output files:

```text
runs/fno3d_baseline/
  metrics/test_summary.json
  metrics/test_per_timestep.csv
  predictions/test/prediction_tXXXX.npy
```

- `prediction_tXXXX.npy`: correction only, `(Z,Y,X,3)`, like the 3D U-Net project. To reconstruct DNS, add to the corresponding `input_tXXXX.npy`.
- `reconstruction_mse`: MSE comparing `input + correction_prediction` to `input + ground_truth_correction` (smaller is better).
- `baseline_mse`: MSE of the input alone (without any model) to DNS. Your trained model should ideally improve upon this.
- `reconstruction_rel_l2`: relative norm of the full-field reconstruction error (smaller is better).
- `correction_mse`: MSE of predicted versus true correction.
- `correction_rel_l2`: relative correction error. Can be unstable when the true correction norm is tiny.

The script also reports the configured loss components and per-timestep rows. The output is in **physical original data units** if `data.scale` is provided consistently; using `scale: 1.0` keeps parity with the repo.

## 8. Hyperparameters you can tune

**Model capacity**

| Setting | Initial value | Values to test | Meaning / trade-off |
|---|---:|---|---|
| `model.width` | 16 | 8, 16, 24, 32 | Number of learned feature channels. Larger -> more expressive but much more memory/compute. |
| `model.modes` | `[8,8,8]` | `[6,6,6]`, `[8,8,8]`, `[12,12,12]` | Number of 3D Fourier modes retained. Larger -> more frequency detail but spectral weights grow quickly. |
| `model.depth` | 3 | 2, 3, 4, 5 | Number of hybrid FNO layers. More may learn complex interactions but cost more. |
| `model.local_kernel` | 3 | 1, 3, 5 | Size of local spatial convolution receptive field. |
| `model.local_path` | true | true, false | Ablation: test whether local convolutions improve the FNO. |
| `model.residual_blocks` | true | true, false | Ablation: whether each block adds its input features before GELU. |
| `model.coordinates` | false | false, true | Add normalized XYZ coordinates; may hurt homogeneous translation-invariant flows, but test if location matters. |

**Training behavior**

| Setting | Initial value | Values to test | Meaning / trade-off |
|---|---:|---|---|
| `training.lr` | 0.0005 | 0.001, 0.0005, 0.0001 | Learning rate; too large can diverge; too small converges slowly. |
| `training.batch_size` | 1 | 1, 2 (if VRAM allows) | Examples processed at once. |
| `training.epochs` | 100 | 50, 100, 200 | Maximum training passes. Monitor validation and early stopping. |
| `training.weight_decay` | 0 | 0, 1e-5, 1e-4 | Regularization of model weights. |
| `training.grad_accum_steps` | 1 | 1, 2, 4 | Simulates a larger batch without storing all examples at once. |
| `training.amp` | false | false, true (CUDA) | Mixed precision may save memory/time, but FFTs still run in float32. |
| `data.train_patch_size` | null | null, 64, 96 | Train on subcubes if full 128³ does not fit. **Not equivalent** to full-cube Fourier training; see warning below. |

**Loss function**

| Setting | Initial value | Values to test | Meaning / trade-off |
|---|---:|---|---|
| `loss.pixel_weight` | 1.0 | keep 1.0 initially | Emphasize individual velocity values, via MSE of corrections. |
| `loss.gradient_weight` | 0.1 | 0, 0.05, 0.1, 0.2 | Encourage correct spatial changes in the velocity field. Matches the existing 3D U-Net's starting gradient weight. |
| `loss.spectral_weight` | 0.0 | 0, 0.001, 0.01 | Penalize incorrect **Fourier magnitudes of reconstructed fields**. This is **not** the isotropic kinetic energy spectrum; additional 3D FFT memory/compute. |
| `loss.divergence_weight` | 0.0 | 0 initially, small values after checks | Optional incompressibility encouragement for periodic vector fields. Use physically correct `spacing: [dz,dy,dx]`. |
| `loss.periodic` | true | true / false | Periodic forward differences for gradient loss. Use false if data boundaries are not periodic. |

`data.scale` divides **both** inputs and corrections by the same constant; keep at 1.0 for the cleanest comparison to the current U-Net. Any other positive value changes loss magnitudes and may require retuning.

**Avoid changing many parameters at once.** Change one feature (such as width or modes), keep the data split and loss unchanged, compare on validation, then try the next experiment.

## 9. Quick command-line experiments

Run each with a different name. For example:

```bash
# Higher capacity: 24 feature channels
python train.py -o data.processed_dir=/absolute/path/to/processed/cnn \
  -o experiment.run_name=fno3d_width24 -o model.width=24

# Higher frequency capacity: 12 modes per spatial dimension
python train.py -o data.processed_dir=/absolute/path/to/processed/cnn \
  -o experiment.run_name=fno3d_modes12 -o model.modes='[12,12,12]'

# More layers
python train.py -o data.processed_dir=/absolute/path/to/processed/cnn \
  -o experiment.run_name=fno3d_depth5 -o model.depth=5

# Optional spectral magnitude loss (not kinetic energy spectrum)
python train.py -o data.processed_dir=/absolute/path/to/processed/cnn \
  -o experiment.run_name=fno3d_spec001 -o loss.spectral_weight=0.001

# Optional small-cube training if full 128^3 is out of GPU memory
python train.py -o data.processed_dir=/absolute/path/to/processed/cnn \
  -o experiment.run_name=fno3d_patch64 -o data.train_patch_size=64 \
  -o loss.periodic=false
```

Or run the **validation-only 5-trial sweep**, sequentially (one trial at a time):

```bash
python sweep.py --data-dir /absolute/path/to/processed/cnn --dry-run
python sweep.py --data-dir /absolute/path/to/processed/cnn
```

The sweep writes `runs/sweep_fno3d_validation_ranking.csv`; it does not inspect the test split. If your GPU cannot fit the `width=24` or `depth=5` candidates, reduce the sweep's candidate sizes, use AMP, or reduce patch size **consistently across trials**.

## 10. Fair comparison with the existing 3D U-Net

Your repo's configuration is `configs/sr_unet_meteo_3d_velocity128.yaml`. For a meaningful controlled comparison:

1. Use the **exact same `data/processed/cnn` train/val/test files**. Keep the same timestep split: do not move samples between splits.
2. Use identical input/target definitions: `input` is interpolated velocity and `target=DNS-input` is the correction. Both FNO and U-Net predict corrections.
3. Start with identical normalization (`none`, here `data.scale: 1.0`), augmentation (off), loss (pixel L2 1.0 + gradient 0.1 with periodic differences), training epochs (100), AdamW and seed (42) when possible.
4. Choose the best FNO checkpoint from **validation** and test **once** after the model and hyperparameters are selected.
5. Report not just correction MSE but **full-field reconstruction MSE / rel-L2**, improvement over interpolated baseline, parameter count, GPU memory, training time, and inference time.
6. Parameter-count-matched and compute-matched comparisons can add rigor: an FNO with much fewer parameters should not automatically be declared worse or better solely based on accuracy.

**Caveat:** no architecture alone guarantees a divergence-free reconstruction, correct small-scale turbulent energy spectra, or better SR performance. Visualize 3D flow structures and measure physically meaningful quantities alongside loss metrics.

## 11. Important 3D / FNO considerations

- **3D FNOs are memory-intensive.** On 128³ cubes, 3D FFTs and feature activations may require substantial GPU memory. Start with `width=16`, `modes=[8,8,8]`, `depth=3`, `batch_size=1`, then reduce width/depth if OOM. The project does not guarantee a specific VRAM requirement.
- **Fourier mode limits:** for a spatial dimension `(D,H,W)`, the first two mode counts must be at most `floor(D/2)` and `floor(H/2)` to prevent overlap; the third must be at most `floor(W/2)+1`. More modes increase parameter cost roughly with their product.
- **Periodicity:** FFT-based mixing implicitly treats the cube as periodic. If extracted JHTDB cubes are **subvolumes rather than complete periodic boxes**, opposite faces need not match. The optional circular-padding local conv and periodic gradient loss can then produce boundary artifacts. Do not assume periodicity merely because the parent DNS simulation is periodic; inspect the exact extracted subvolume and consider `loss.periodic=false` or an alternative boundary treatment.
- **Patch training:** cropping 64³ from a 128³ volume reduces memory, but each cropped patch is treated by the Fourier layer as its own domain. This changes the frequency interpretation and may create false periodic boundaries. Full-cube training/evaluation is preferable when practical. Validation always uses **uncropped** cubes. The trained model can accept different spatial shapes as long as Fourier mode limits are satisfied, but equivalence across grid sizes is not guaranteed.
- **The loss's optional divergence term** assumes `(u,v,w)` channels and `(Z,Y,X)` spatial axes, with spacing `[dz,dy,dx]`. Verify this arrangement and the physical grid spacings before turning it on.
- **AMP and FFT:** the spectral FFT is intentionally computed in FP32 to prevent low-precision FFT failures on some CUDA devices. `training.amp=true` affects other layers but will not make the Fourier transforms half-precision.
- **No test tuning:** the `test/` folder is read only in explicit `--evaluate --split test` runs, never for training/sweep.
- **No data download:** this project loads existing processed files; it doesn't invoke JHTDB APIs or modify the source repository.

## 12. Troubleshooting

- `Set data.processed_dir`: supply the *directory with `train/ val/ test/`* using `-o data.processed_dir=/absolute/.../cnn`.
- `No matched pairs`: check filenames follow `input_tXXXX.npy` / `target_tXXXX.npy` and both exist for each timestep.
- `Fourier modes exceed limits`: reduce `model.modes` or increase spatial cube size; for 16³ synthetic data try `[3,3,3]`.
- CUDA out of memory: decrease width/depth, use batch size 1, try AMP, or use patch training with its caveats. Evaluation still runs on full cubes.
- Run already exists: choose a new `experiment.run_name`, or use `--resume` with an existing trusted checkpoint. Nothing is silently overwritten by a new training run.
- Training loss explodes: disable AMP, reduce LR to `1e-4`, verify target scaling and array orientation, and inspect data for extreme values.
- CPU too slow: full 128³ training should be run on a suitable GPU allocation.

## 13. What was tested

- Five Python unit tests: zero-initialized baseline, 3D tensor shape, FFT backpropagation, mode validation, dataset pairs, and spectral/gradient loss backprop.
- End-to-end **synthetic** 16³ train (two epochs), checkpoint saving, best model selection, test evaluation, and predicted correction `.npy` export, on PyTorch CPU.
- **Not performed:** training on JHTDB 128³ velocity cubes, GPU memory profiling, scientific validation, and comparison against existing U-Net checkpoints. Those require your actual dataset, compute allocation and evaluation procedure.

### Source repository inspected (read-only)

[GitHub: nitipon-ton/jhtdb-deep-learning-vip](https://github.com/nitipon-ton/jhtdb-deep-learning-vip), especially `README.md`, `configs/sr_unet_meteo_3d_velocity128.yaml`, `utils/cube_datasets.py`, `utils/training_3d.py`, `scripts/train_3d.py` and `models/SR/unet_meteo_3d.py`.


## Repository integration notes

- **Location:** `models/SR/fno_3d/`; the existing `models/SR/unet_meteo_3d.py` is unchanged.
- **Run from repo root:** `python models/SR/fno_3d/train.py -o data.processed_dir=/absolute/path/to/processed/cnn -o experiment.run_name=fno3d_baseline`
- **Import model from repo root:** `from models.SR.fno_3d import HybridFNO3D`.
- Training outputs default to `models/SR/fno_3d/runs/` and are excluded from Git by the FNO-local `.gitignore`.
- No existing model registry, shared training utility, or U-Net configuration is edited.
- Never train on `test` for hyperparameter selection; test only after selecting on `val`.
