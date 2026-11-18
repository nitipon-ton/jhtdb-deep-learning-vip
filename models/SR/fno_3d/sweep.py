#!/usr/bin/env python3
"""Small, sequential validation-only hyperparameter sweep (no test evaluation).

Run from this directory with:
  python sweep.py --data-dir /absolute/path/to/processed/cnn --dry-run
  python sweep.py --data-dir /absolute/path/to/processed/cnn

Each trial uses a distinct run name and saves its own checkpoints. This script does not change existing U-Net files and never opens the test split.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Deliberately small study. Values are starting hypotheses, NOT claimed optima.
TRIALS = [
    dict(label='baseline', width=16, modes=[8,8,8], depth=3, lr=5e-4),
    dict(label='wide', width=24, modes=[8,8,8], depth=3, lr=5e-4),
    dict(label='more_modes', width=16, modes=[12,12,12], depth=3, lr=5e-4),
    dict(label='deeper', width=16, modes=[8,8,8], depth=5, lr=5e-4),
    dict(label='lower_lr', width=16, modes=[8,8,8], depth=3, lr=1e-4),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True,
                        help='Absolute path to processed/cnn containing train val test')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--name-prefix', default='sweep_fno3d')
    parser.add_argument('--config', type=Path, default=HERE/'config.yaml')
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error('--epochs must be positive')
    rows = []
    for trial in TRIALS:
        name = f'{args.name_prefix}_{trial["label"]}'
        command = [sys.executable, str(HERE/'train.py'), '--config', str(args.config),
                   '-o', f'data.processed_dir={args.data_dir}',
                   '-o', f'experiment.run_name={name}',
                   '-o', f'training.epochs={args.epochs}',
                   '-o', f'model.width={trial["width"]}',
                   '-o', 'model.modes='+str(trial['modes']).replace(' ', ''),
                   '-o', f'model.depth={trial["depth"]}',
                   '-o', f'training.lr={trial["lr"]}']
        print(' '.join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, check=True)
        history_path=HERE/'runs'/name/'history.csv'
        # If custom output_dir is used, prefer explicit direct training CLI instead.
        if not history_path.exists():
            raise FileNotFoundError(f'Expected history at {history_path}; '
                                    'sweep.py assumes experiment.output_dir=runs')
        with history_path.open(newline='') as f:
            hist=list(csv.DictReader(f))
        best=min(hist, key=lambda r: float(r['val_total']))
        rows.append({**trial, 'run_name': name, 'best_val_total': float(best['val_total']),
                     'best_epoch': int(best['epoch'])})
    if rows:
        rows.sort(key=lambda x: x['best_val_total'])
        output=HERE/'runs'/f'{args.name_prefix}_validation_ranking.csv'
        with output.open('w',newline='') as f:
            cols=['run_name','best_val_total','best_epoch','width','modes','depth','lr','label']
            writer=csv.DictWriter(f,fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
        print(f'Validation-only ranking: {output}')
        print(f'Best trial by minimum validation objective: {rows[0]["run_name"]}')
        print('Do not examine the held-out test split until after selecting the model.')


if __name__ == '__main__':
    main()
