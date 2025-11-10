# JHTDB-ML

Adapted from and built on top of the excellent [pyJHTDB](https://github.com/idies/pyJHTDB) project. This repository adds a practical workflow for fetching channel-flow planes from the Johns Hopkins Turbulence Database (JHTDB) and for experimenting with data-driven and machine learning models like super-resolution models on the resulting datasets. Maintained by [Aditya Sai Pranith Ayapilla](https://sites.google.com/view/adityasaipranithayapilla/home). For more information on the Johns Hopkins turbulence database, please check https://turbulence.idies.jhu.edu/database.

---

## Core capabilities

1. **Data acquisition and dataset generation**  
   `notebooks/jhtdb_data_fetching.ipynb` connects to JHTDB, respects query limits, tiles the velocity field, and saves consistent snapshot bundles to `data/raw/`.

2. **Macine learning pipeline**  
   `notebooks/ml_pipeline_sr.ipynb` consumes those snapshots, prepares training tensors, instantiates registered models (currently the Hybrid DSC/MS variant), and evaluates the predictions with full-field patch blending and rich visual diagnostics.

More data-driven frameworks (e.g., modal decomposition, denoising) will be layered on top of this foundation.

---

## 1) Data fetching and preprocessing

### Notebook: `notebooks/jhtdb_data_fetching.ipynb`

This workflow is designed to be rerun whenever new snapshots are needed:

1. **Configuration** – set the flow case, pick the wall-normal location, the number of snapshots, and the temporal spacing. The notebook references the same YAML configuration structure used elsewhere so runs remain reproducible.
2. **Authentication** – supply your JHTDB API token (the notebook reads it from `.env` for convenience).
3. **Tiled acquisition** – JHTDB imposes a 4096-point limit per request. The notebook decomposes the target plane into manageable tiles, fetches each tile, and stitches them into a full 256×256 field for every snapshot. All three velocity components plus derived quantities such as vorticity are retrieved. Of course, you can get your own API key to access more than 4096 points per query. Check https://turbulence.idies.jhu.edu/database
4. **Quality checks** – visual comparisons and basic statistics ensure the downloads look sensible before committing them to disk.
5. **Serialisation** – every snapshot bundle is written to `data/raw/` using portable HDF5 files. The downstream SR notebook expects files shaped `(time, component, height, width)` with consistent naming, so once this notebook completes you can move straight to model training without additional massaging.

This data-fetch notebook is intentionally self-contained: each cell logs progress, catches API throttling, and saves intermediate tiles to avoid repeating expensive queries.

---

## 2) Machine Learning pipeline

### What’s implemented right now

- **Data ingestion from local HDF5** created by the fetch notebook: loading velocity components and derived fields, global normalisation, and synthetic low-resolution creation through configurable downsampling.
- **Patch extraction with augmentation** (flips, rotations, optional noise) to build paired LR/HR tensors ready for PyTorch training.
- **Model zoo** focused on super-resolution: the primary entry point builds `HybridDSCMSR`, a wrapper around the hybrid DSC/MS architecture that handles scale-aware upsampling. Other registered builders (plain U-Net, small MLPs, etc.) remain available through the factory.
- **Training loop inside `notebooks/ml_pipeline_sr.ipynb`** with metric logging, optional gradient clipping, LR scheduling, and an automatic guard that disables mixed precision when running on Apple MPS to avoid kernel crashes.
- **Full-field evaluation** that tiles a 64×64 low-resolution plane into overlapping patches, runs the network on each patch, and blends the 64×64 predictions into a seamless 256×256 output using a Hann window. Baseline interpolations (nearest, bilinear, bicubic) are generated for side-by-side comparisons.
- **Metric and figure generation**: PSNR, MAE, correlation tables plus profile plots and contour maps saved into the experiment workspace.

---

### Repository layout

```
├── configs/              # YAML experiment settings
├── dataloaders/          # (archived under backups/) legacy CLI fetchers
├── guides/               # Short developer notes
├── models/               # Registered architectures and wrappers
├── notebooks/
│   ├── jhtdb_data_fetching.ipynb  # Snapshot acquisition
│   └── ml_pipeline_sr.ipynb      # End-to-end SR workflow
├── utils/                # Metrics, physics helpers, IO utilities
├── work/                 # Auto-created run directories (checkpoints, plots)
└── README.md
```

Everything in `work/`, `data/`, and `saved_models/` is ignored by git so local experiments stay isolated; `.gitkeep` files preserve the folder structure.

---

### Getting started

1. **Environment**  
   ```bash
   conda env create -f environment.yml
   conda activate jhtdb-ml
   ```

2. **Data**  
   Place your channel-flow HDF5 file (e.g. `data/channel_sr_256.h5`) in the `data/` directory. The SR notebook expects fields named `ux`, `uy`, `uz`, `omega_z`, `time`, `y_value` with 256×256 slices.

3. **Run the pipeline notebook**  
   ```bash
   jupyter notebook notebooks/02_sr_pipeline.ipynb
   ```
   The notebook walks through configuration loading, data inspection, patch generation, model instantiation, training, and evaluation. All intermediate artefacts are stored in a timestamped folder under `work/SR/`.

---

### Training workflow in the notebook

1. **Configuration** – reads a YAML experiment file, resolves paths, and prints the active hyperparameters.
2. **Data preparation** – loads all snapshots, normalises them (z-score), creates synthetic LR fields by average pooling, and extracts overlapped LR/HR patch pairs according to the scale factor.
3. **Model selection** – builds the configured architecture via `models.factory.build_model_from_config`. Any new model can be registered without modifying the notebook.
4. **Training loop** – runs for the requested epochs with progress bars for train/val splits, logs PSNR, saves the best checkpoint, and serialises the history to disk.
5. **Patch-based reconstruction** – after training, predicts SR patches for the full LR plane, blends them into a 256×256 field, and compares against interpolation baselines using consistent colour limits and statistics.

Mixed precision is automatically disabled when the active device is Apple’s MPS backend, preventing the common fp16 kernel crash on macOS.

---


### Switching models or experiments

- Update the YAML under `configs/` (e.g. change `model.architecture` to `unet` or tweak training parameters).  
- For scripted runs outside the notebook, `models.factory.available_models()` lists the registered architectures, and the helpers in `utils/experiment_manager.py` keep work directories tidy.

---

### Troubleshooting tips

- Kernel crash on macOS: leave mixed precision off (default behaviour now).  
- Memory pressure: reduce `training.batch_size` or use fewer snapshots.  
- Slow data loading: keep `num_workers=0` inside the notebook; parallel loading can be re-enabled cautiously in scripted runs.  
- HDF5 key mismatch: inspect available datasets with `h5py.File` and adjust the config or loader logic accordingly.

---

## License

This project is released under the MIT License. See `LICENSE` for full text.
