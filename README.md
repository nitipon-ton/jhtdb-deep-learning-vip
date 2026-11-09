# JHTDB-DEEP-LEARNING

A repository to explore and test data-driven and deep learning frameworks on fluid dynamics datasets - provided by the [Johns Hopkins Turbulence Database](https://turbulence.idies.jhu.edu/home). 
Adapted from and built on top of the excellent [pyJHTDB](https://github.com/idies/pyJHTDB) project. This repository adds a practical workflow for fetching data from the Johns Hopkins Turbulence Database (JHTDB) and for experimenting with data-driven and machine learning models like super-resolution models on the resulting datasets. Maintained by [Aditya Sai Pranith Ayapilla](https://sites.google.com/view/adityasaipranithayapilla/home). For more information on the Johns Hopkins turbulence database, please check https://turbulence.idies.jhu.edu/database.

---

## Core capabilities

1. **Data acquisition and dataset generation**  
   `notebooks/jhtdb/jhtdb_data_fetching.ipynb` connects to JHTDB, respects query limits, tiles the desired field data, and saves consistent snapshot bundles to `data/raw/`.

2. **Macine learning pipeline**  
   `notebooks/super_resolution/data_inspection.ipynb`, `notebooks/super_resolution/training_sr.ipynb`, and `notebooks/super_resolution/postprocessing_sr.ipynb` consume those datasets, prepare training tensors, instantiate registered models, train them (optionally with physics-informed losses), and evaluate predictions with both image-based and physics-based diagnostics.

More data-driven frameworks (e.g., modal decomposition, denoising) will be layered on top of this foundation.

---

## 1) Data fetching and preprocessing

### Notebooks under `notebooks/jhtdb/`

- **`jhtdb_data_fetching.ipynb`**  
  - Configure the flow case and query settings via `configs_jhtdb/*.yaml`.  
  - Fetch snapshots from JHTDB, tile requests to respect point limits, and stitch tiles into full planes with fields like `ux`, `uy`, `uz`, `omega_z`, and `time`.  
  - Store everything as HDF5 under `data/raw/<case>/` and perform quick visual checks before heavy training.

- **`prepare_isotropic_xy_128.ipynb`**  
  - Load `data/raw/isotropic_xy_512/isotropic_xy_512.h5`.  
  - Split each 512×512 snapshot into 128×128 tiles and write them to `data/raw/isotropic_xy_128/isotropic_xy_128.h5`.  
  - This 128×128 dataset is used by the SR notebooks for 4× upscaling from 32×32 LR.

If you already have compatible HDF5 files, you can point the SR configs directly at your own data and skip these steps.

---

## 2) Machine Learning pipeline

### What’s implemented right now

- **Config-driven experiments**  
  - YAML configs in `configs/` define dataset paths, splits, model architecture, training hyperparameters, and physics-loss coefficients.  
  - The notebooks read one config and wire datasets, models, and training via `models.factory.build_model_from_config`.

- **SR models**  
  - Meteorology-inspired UNet (`unet_meteo`) for 4× SR (e.g. 32→128 or 128→512) following Yasuda et al. (2023). 
  - Hybrid DSC/MS model (`hdsc_ms`) following Fukami et al. (2019).  
  - EDSR variants and an MLP baseline.  
  All assume single-channel SR (ω_z), with optional temporal stacking handled by the dataset classes.


- **Training and evaluation notebooks**  
  - `data_inspection.ipynb`: inspect ranges, normalisation, and several snapshots before training.  
  - `training_sr.ipynb`: main training loop with multi-GPU (DataParallel), optional mixed precision (CUDA only), gradient clipping, LR schedulers, and structured run directories under `work/SR/`.  
  - `postprocessing_sr.ipynb`: evaluate a chosen run on test data, generate full-field and zoom comparisons (bicubic vs SR vs HR), error maps, PDFs, energy spectra, and physics-term comparisons.
  - [`postprocessing_sr_comparison.ipynb`](notebooks/super_resolution/postprocessing_sr_comparison.ipynb): controlled 32×32 → 128×128 omega_z comparison. On PACE, use the project environment in a GPU allocation, edit the first settings cell, and Run All. Defaults to the corrected U-Net and residual CNN configs; add compatible configs to `CONFIG_PATHS` to compare more architectures. Requires each exact run's validation-selected `checkpoints/best_model.pth` and evaluates all 400 held-out snapshots against bicubic. Saves JSON provenance, aggregate/per-sample CSVs, and comparison plots under a new timestamped `work/SR/comparisons/` directory. Reports physical-unit pixel metrics, training-compatible physics errors, additional physical derivative errors, parameter counts, and synchronized FP32 batch-one forward timings. Test results are reporting only; model and epoch selection must use validation data. Evaluation helpers live in `utils/sr_comparison.py`; run regression checks in the project environment with `python -m unittest test_scripts.test_sr_comparison`.

### Repository layout (relevant pieces)

```
configs/             # SR experiment configs (models, training, data paths)
configs_jhtdb/       # JHTDB fetch configs (API and plane selection)
data/raw/            # HDF5 datasets (fetched + preprocessed)
models/SR/           # SR architectures (UNet, hDSC/MS, EDSR, MLP, etc.)
models/factory.py    # Model registry and build helpers
notebooks/jhtdb/     # Data fetching and tiling notebooks
notebooks/super_resolution/
  ├─ data_inspection.ipynb
  ├─ training_sr.ipynb
  └─ postprocessing_sr.ipynb
utils/               # Experiment manager, metrics, physics utilities
work/SR/             # Auto-created run directories (configs, checkpoints, plots)
```
---

### Switching models or experiments

- Update the YAML under `configs/` (e.g. change `model.architecture` to `unet` or tweak training parameters).  
- For scripted runs outside the notebook, `models.factory.available_models()` lists the registered architectures, and the helpers in `utils/experiment_manager.py` keep work directories tidy.

---

### Create Environment to for Getting started (or run in a docker container - see below.)

```bash
conda env create -f environment.yml
conda activate jhtdb-ml
```

---

## Docker / Containers

- A public Docker setup is provided under `Docker/`:
  - `Docker/Dockerfile`: builds a Miniconda-based image using `environment.yml`.
  - `Docker/docker-compose.yml`: optional Compose file for running the project (e.g., with Jupyter Lab).
- It is generally recommended to run this repository inside a container for reproducibility and clean isolation from host dependencies.
- If you use VS Code, you can also open the repo in a Dev Container using the “Dev Containers” extension; you may adapt the container configuration to your own environment as needed.

---

## License

This project is released under the MIT License. See `LICENSE` for full text.
