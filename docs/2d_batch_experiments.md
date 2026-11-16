# Automated larger-model 2D experiments

The batch runner replaces manually selecting and executing a notebook for each
candidate. It trains sequentially on one allocated GPU, isolates each trial in
a new Python process, records errors, and writes validation-only results after
each trial. It never evaluates test arrays or stops based on a test-MSE target.

## One command on PACE

Upload/pull the updated `scripts/`, `utils/`, `configs/study_2d/` and comparison
notebook. Keep old `data/` and `work/`. Use the project Python environment in an
allocated GPU session/job, with enough wall time for the declared search:

```bash
cd ~/scratch/jhtdb-deep-learning-vip
python -u scripts/run_2d_experiments.py --plan configs/study_2d/large_search.yaml
```

No separate prepare step or notebook clicks are required. This does not request
a PACE GPU or submit a Slurm job for you; the command runs inside your existing
allocation. Its lifetime is limited by that allocation and its parent terminal
or job. Prefer an allocated batch job for a long unattended search.

Optional preview (writes the frozen configs/manifest, but does not train):

```bash
python scripts/run_2d_experiments.py --plan configs/study_2d/large_search.yaml --dry-run
```

A first pilot can train only the first pending trial, with all epoch logs live:

```bash
python -u scripts/run_2d_experiments.py --plan configs/study_2d/large_search.yaml --limit 1 --verbose
```

Then run the original command to finish the remaining trials. Completed runs
are skipped only after checking saved config, timing completion, best checkpoint
hash and selection metadata. A pilot result is not a complete search result.

## What the new search changes

`configs/study_2d/large_search.yaml` declares **96 trials: 12 per family across all
eight families**, including CNN. It uses reproducible random sampling without
duplicate model/training configurations within a family. The first three
trials cover the three declared capacities; later trials sample capacities and
other settings. Equal trial count is not equal compute cost.

- Larger/deeper architectures: up to about 48.5M parameters for meteorology
  U-Net, 17.3M compact U-Net, 10.8M EDSR, 1.17M HDSC/MS, 2.94M SwinIR Lite,
  18.1M RRDB generator, 3.55M residual CNN and 52.5M MLP (largest declared
  capacities under base options). These are measured parameter counts, not
  performance predictions. The selected random options can change the count.
- Learning rates 1e-4/3e-4/6e-4/1e-3 and weight decay 0/1e-5/1e-4.
- Additional model options declared in the YAML, such as CNN normalization,
  EDSR residual scaling and MLP dropout. MLP remains the direct-output baseline;
  no residual shortcut is silently added. RRDB still has adversarial loss off.
- Up to 300 epochs, at least 75 epochs before early stopping, patience 50.
- ReduceLROnPlateau halves learning rate after validation stalls, with patience
  8. This gives lower learning rates a chance before early stopping, unlike the
  previous long cosine schedule with patience 20.
- Microbatch 16, accumulation 2: effective batch up to 32, with the final partial
  group handled by actual sample count. BatchNorm still sees microbatches of 16;
  accumulation does not reproduce BatchNorm statistics from a batch of 32.

The corrected composite **training** loss, normalization, dataset, and split
remain fixed. Because this round explicitly targets MSE, the declared
`training.selection_metric` is now **`val_mse`**: physical-unit validation
omega_z MSE. Checkpoint selection, scheduling and early stopping all use that
metric. Composite validation loss is logged too. These checkpoints are selected
differently from the first sweep's minimum composite-loss checkpoints; disclose
that change. To run a separate composite-selected experiment, choose a new tag
and output directory and set `selection_metric: val_loss` (and early stopping
metric consistently) before generating its candidates.

Larger networks and a broader finite search do not guarantee test MSE below
50. Validation MSE and test MSE are different measurements. The existing test
set has already been inspected and influenced development, so expanded results
on it should be described as exploratory. A new independent holdout is needed
for fresh confirmation. Finalized settings can also be repeated across seeds;
this initial search holds training seed 42 fixed rather than searching seeds.

## Outputs

Each run is saved separately:

```text
work/SR/study2d_large01_<architecture>_tNNN_s42/
  config.yaml
  checkpoints/best_model.pth
  checkpoints/last_model.pth
  logs/training_log.csv           # every loss component, sample counts, val MSE, LR
  logs/epoch_timing.csv
  metrics/training_timing.json    # same schema used by comparison notebook
  metrics/training_summary.json  # declared metric and selected epoch's loss/MSE
  metrics/data_provenance.json   # train/val keys, train-only stats, no test reads
  history/training_history_v1.json
  plots/training_curves.png
```

Batch-level files:

```text
work/SR/searches/large01/
  manifest.json                  # frozen plan and candidate config hashes
  configs/*.yaml                 # exact configs to use for final comparison
  logs/*.log                     # complete stdout/stderr for each attempt
  batch_status.json              # completion/failure state and attempt wall times
  validation_results.csv         # completed runs ordered by declared validation score
  selected_configs.json          # best completed candidate per architecture
  failed_attempts/               # only created by explicit --retry-failed
```

Selections are marked `provisional: true` until all planned candidates complete.
Do not treat an early partial winner as the result of the full search. The batch
report sums completed training times; failed attempts have separate worker wall
times, and should be included when reporting total search cost. Worker wall time
also includes process/data/model setup and plotting, unlike timed training.

The batch training loop shares the repository's model factory, multifield
dataset and physics/gradient loss implementations with the notebook, but is a
separate headless loop. It logs validation pixel MSE explicitly, supports exact
sample-weighted gradient accumulation and a plateau scheduler. It does not
import or execute notebook cells, initialize a discriminator, or load optional
historical notebook weights. Validation loops use the same precision setting
as training; final inference comparisons remain FP32 as before.

## Restart, errors and memory

Re-run the same command to skip verified completed experiments and continue
pending ones. This is **experiment-level restart, not epoch-level resume**.
Allocation loss can leave an incomplete trial; no optimizer/scaler resume is
claimed. Without a retry flag, nonempty incomplete folders are preserved and
reported. Retry them explicitly:

```bash
python -u scripts/run_2d_experiments.py --plan configs/study_2d/large_search.yaml --retry-failed
```

That moves incomplete folders intact into `failed_attempts/` and starts those
trials from scratch. Completed trials are not retrained. Errors, including OOM,
are logged and the next experiment continues; the script does not silently
shrink a batch or change hyperparameters. `--fail-fast` stops at the first
failed/preserved trial instead. Investigate repeated failures before retrying.

A lock prevents two controllers from launching the same search concurrently.
Normal exits/interrupts remove it. After forced termination, inspect `.batch.lock`
and confirm the recorded process is no longer running before removing it.
Do not start concurrent manual training on a run owned by this batch.

If VRAM is insufficient, define a **new** plan/tag/output folder with smaller
microbatches and larger accumulation to maintain the intended effective batch.
Never edit a frozen candidate or overwrite old results. Existing manifests are
checked against their plan and config hashes on every invocation.

The same runner can automate the old 48 generated configs too:

```bash
python -u scripts/run_2d_experiments.py --configs-dir configs/study_2d/sweep_all
```

It will recognize matching completed notebook runs and skip them. This mode
retains each config's original selection metric (default `val_loss`); it does
not apply the new plan's settings. Batches must use a single selection metric
and matching data/domain/loss definitions. The legacy global validation summary
script now groups different selection metrics rather than ranking their scales
together; prefer the batch-specific CSV for this new search.

## Final comparison

Once the batch is complete and choices are frozen, load its selections in the
comparison notebook's settings cell (set your actual PACE absolute path):

```python
import json
selected_path = Path('/absolute/project/path/work/SR/searches/large01/selected_configs.json')
selected = json.loads(selected_path.read_text())
assert not selected['provisional'], 'Finish all planned trials first.'
CONFIG_PATHS = selected['CONFIG_PATHS']
```

Then Run All to compare those validation-selected best checkpoints with bicubic.
The batch runner never performs this test step automatically or repeats test
evaluation until a threshold is reached.

## Checks

In the project environment:

```bash
python -m unittest test_scripts.test_batch_2d test_scripts.test_2d_study test_scripts.test_sr_comparison
```

Synthetic-data checks cover test-array exclusion, actual training and artifact
creation, conflicting validation objectives, exact partial-group accumulation,
search reproducibility, edited-config rejection, skip/failure/archive/retry
behavior, and finite SR outputs for the largest declared capacities. CPU checks
do not establish GPU memory fit, CUDA/AMP behavior, accuracy, or runtime on PACE.

References: [random-search budgeting](https://www.jmlr.org/papers/v13/bergstra12a.html),
[PyTorch ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).
