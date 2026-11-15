# Controlled 2D retraining study

This study uses 32x32 average-pooled input and 128x128 omega_z targets, the
chronological 3000/600/400 split, training-only global normalization, and the
corrected pixel + gradient + advection + diffusion objective.

## Interpretation and fairness

U-Net is itself a CNN. Its encoder/decoder and skip connections provide a
multiscale inductive bias, rather than a guarantee of lower error than a
residual CNN. One run is not an architecture-level superiority result.

- **Shared recipe:** train each reference config under the common settings.
  This measures performance under that recipe, not each model's optimum.
- **Tuned comparison:** declare search spaces and trial/compute budgets first;
  select configurations by validation composite loss, then freeze them before
  test reporting. Equal trial count is not equal GPU time; report both.
- If CNN stays fixed while other families are tuned, label the result
  **tuned alternatives versus a fixed CNN baseline**. Equal tuning effort
  requires including CNN in the budget too.

The existing 400 test samples have already been inspected during development.
Keep them out of further tuning and disclose this history. Retraining does not
make them unseen again; fresh confirmation needs a new independent holdout.

## Configs

All configs below are **starting candidates, not proven optimized settings**.
They cover all eight existing 2D model families in `models/factory.py`.
An additional architecture needs Python implementation; YAML alone cannot add it.

| File in `configs/study_2d/` | Architecture | Starting capacity |
| --- | --- | --- |
| `unet_meteo.yaml` | Current meteorology U-Net | [32,64,128,256,256]; 2 blocks/stage |
| `unet.yaml` | Compact U-Net | 32 base channels |
| `residual_cnn.yaml` | Fixed CNN using Tony's latest supplied settings | 32 channels, 4 blocks, instance normalization, dropout 0.05 |
| `edsr.yaml` | EDSR | 16 blocks, 64 features |
| `hdsc_ms.yaml` | Hybrid DSC/MS | DSC width 32; MS widths 16/8/3 |
| `swinir.yaml` | Repository SwinIR Lite | embedding 64, depth 6, 4 heads, window 8 |
| `esrgan.yaml` | RRDB generator | 8 RRDBs, 64 features, growth 32; GAN disabled |
| `mlp.yaml` | Fully connected baseline | hidden widths [512,1024,512], dropout 0 |

RRDB with the shared composite loss is not a full adversarial ESRGAN experiment.
The SwinIR entry is the repository's Lite implementation, not a reproduction of
the full published training recipe.

Shared defaults: AdamW, lr 1e-4, batch size 32, maximum 200 epochs, early stopping
patience 20 on validation composite loss, seed 42, mixed precision, fixed loss
weights, no augmentation. Cosine `t_max` is now **200**, matching the epoch budget;
older configs used 60 and could raise the learning rate again during 200 epochs.
This is a new recipe: compare new runs together and preserve historical results.
Do not tune composite-loss weights independently when ranking by composite loss.

## Train on PACE

1. Upload/pull the updated code, both notebooks and `configs/study_2d/`. Keep
   existing data and old runs. Use the same GPU model/allocation and single-GPU
   setting for comparable costs.
2. Open `notebooks/super_resolution/training_sr.ipynb` in the project environment
   in a GPU allocation. Select a config:

   ```python
   CONFIG_PATH = ROOT / 'configs/study_2d/unet_meteo.yaml'
   ```

3. Restart the kernel and Run All; keep `LOAD_SAVED_MODEL = False`. Study configs
   reject nonempty existing run folders. To repeat, copy the config and assign
   new experiment `name`/`run_name` and the intended `training.seed`.
4. Repeat for other selected configs. Training does not score test data.

Runs save to `work/SR/study2d_<architecture>_reference_s42/`. Creating these local
files does not launch remote training jobs.

## Validation tuning

From the project root in the project environment:

```bash
python scripts/prepare_2d_sweep.py
```

This writes a manifest and **42 YAMLs** to `configs/study_2d/sweep/`: seven
non-CNN families x two capacities x three learning rates (5e-5, 1e-4, 3e-4).
It does not start jobs. Run candidates by selecting their config in the training
notebook. A smaller pilot can run just one family first. Capacity changes may
involve multiple settings; this is a search, not a one-factor causal ablation.

To include CNN for equal trial count, use this instead:

```bash
python scripts/prepare_2d_sweep.py --include-cnn \
    --output-dir configs/study_2d/sweep_all --tag all01
```

That produces **48 candidates**. Do not run both complete sweeps unless you
intend to pay for both. CNN search varies width while keeping instance norm
and dropout fixed. These finite grids cannot establish absolute best capability.
Change `--tag` for a new sweep so run names stay unique.

Summarize completed runs using validation and training timing only:

```bash
python scripts/summarize_2d_validation.py
```

This writes `work/SR/validation_summary.csv`, sorted by validation loss. Filter
by architecture/run name to select a candidate for each family. Reference runs
also appear: include them in the declared search budget if they inform selection.

After selection, repeat the selected settings with seeds 42/43/44 and unique
run names. Keep all seed results; do not pick a seed by test performance. Report
individual runs and mean/spread. The comparison notebook exports individual
runs; it does not calculate cross-seed significance tests or confidence intervals.

## Timing definitions

GPU synchronization occurs at phase boundaries:

- `train_seconds`: training phase including loader wait, transfers,
  forward/backward, optimizer and training progress updates.
- `validation_seconds`: validation phase without weight updates.
- `epoch_seconds`: train + validation + epoch logging/checkpoint writes and
  scheduler work. Excludes the timing report's own disk write.
- `total_training_seconds`: complete timed training-loop session, including
  validation, checkpointing, logging and timing-report writes. Ends after final
  history/last-checkpoint saving. Excludes initial data/model setup, subsequent
  plots/exports/test evaluation and notebook idle time.
- `time_to_best_seconds`: sum of whole epoch times through the validation-selected
  best epoch; excludes between-epoch timing-report writes.

Each run saves:

```text
metrics/training_timing.json       # per-epoch records, summary, device/precision,
                                  # completed flag and best checkpoint hash
logs/epoch_timing.csv              # timings and train/val loss per epoch
metrics/training_summary.json      # existing metadata plus timing summary
history/training_history_v*.json   # loss and per-epoch timing arrays
checkpoints/best_model.pth         # validation-selected model
checkpoints/last_model.pth         # final epoch model
```

Reports persist after each epoch. Interrupted/failed sessions remain
`completed: false` and are not presented as complete training costs. This study
requires fresh training; the old optional weights-only loader is not a full
optimizer resume and is rejected here. Counts can differ due to early stopping;
compare total session cost, per-epoch time, and time to best together.

Validation loss is now weighted by sample count, including the partial last
batch; it may differ slightly from historical unweighted batch averages.
GPU names/count, batch size, precision and seed are recorded. Report costs of
**all trials** as well as final-model cost. Wall time does not measure PACE queue
time, billed GPU hours, power consumption or utilization.

## Final test reporting

In `postprocessing_sr_comparison.ipynb`, set `CONFIG_PATHS` to only the completed,
finalized configs (the selected sweep paths if tuning was performed). Defaults
list all eight reference configs; remove unfinished ones before Run All.

All 400 test samples are evaluated against bicubic with the existing image and
physics metrics, parameter counts and synchronized FP32 inference timing.
Training cost is added to `summary.csv`, `results.json` and
`training_cost_comparison.png`. Historical timings remain null/NaN rather than
zero. A completed timing report must match the selected checkpoint hash.
Full per-epoch records/device information are embedded in each run's JSON
provenance. Compare hardware and batch settings before interpreting speed gaps.

Background: [U-Net architecture](https://arxiv.org/abs/1505.04597) and
[budgeting hyperparameter searches](https://www.jmlr.org/papers/v13/bergstra12a.html).
