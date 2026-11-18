#!/usr/bin/env python3
"""Train/evaluate a standalone 3D hybrid FNO on repo-compatible velocity pairs.

Examples:
 python train.py -o data.processed_dir=/absolute/path/to/processed/cnn
 python train.py --evaluate --split test -o data.processed_dir=/absolute/path/to/processed/cnn
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import VelocityCubePairs
from losses import ResidualLoss
from model import HybridFNO3D

HERE = Path(__file__).resolve().parent


def load_config(path: Path, overrides: list[str]) -> dict:
    with path.open(encoding='utf8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError('Config must be a YAML dictionary')
    for ov in overrides:
        if '=' not in ov:
            raise ValueError(f'Invalid override {ov!r}; expected section.key=value')
        key, raw = ov.split('=', 1)
        cur = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            if part not in cur or not isinstance(cur[part], dict):
                raise ValueError(f'Unknown override key: {key}')
            cur = cur[part]
        if parts[-1] not in cur:
            raise ValueError(f'Unknown override key: {key}')
        cur[parts[-1]] = yaml.safe_load(raw)
    return cfg


def configure_device(requested: str) -> torch.device:
    if requested == 'auto':
        requested = ('cuda' if torch.cuda.is_available() else
                     'mps' if torch.backends.mps.is_available() else 'cpu')
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but not available')
    return torch.device(requested)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(cfg: dict, split: str, train: bool = False):
    dc = cfg['data']
    if not dc.get('processed_dir'):
        raise ValueError('Set data.processed_dir to the folder containing train/ val/ test/')
    return VelocityCubePairs(
        dc['processed_dir'], dc['splits'][split], dc['scale'],
        patch_size=dc.get('train_patch_size') if train else None,
        augmentation=bool(dc.get('augmentation', False)) if train else False,
    )


def dataloader(cfg: dict, split: str, train: bool = False):
    ds = build_dataset(cfg, split, train=train)
    tc = cfg['training']
    return DataLoader(
        ds, batch_size=int(tc['batch_size']), shuffle=train,
        num_workers=int(tc['num_workers']),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(tc['num_workers']) > 0,
        drop_last=False,
    )


def run_dir(cfg: dict) -> Path:
    output = Path(cfg['experiment']['output_dir']).expanduser()
    if not output.is_absolute():
        output = HERE / output
    return output / str(cfg['experiment']['run_name'])


def checkpoint_save(path: Path, *, model, optimizer, scheduler, scaler,
                    cfg, epoch, best_val, best_epoch, stale_epochs):
    state = {
        'epoch': epoch, 'best_val': best_val, 'best_epoch': best_epoch,
        'stale_epochs': stale_epochs, 'config': cfg,
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'scaler': scaler.state_dict() if scaler is not None else None,
    }
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def checkpoint_load(path: Path, device: torch.device):
    # Only load trusted local experiment checkpoints (pickle deserialization).
    return torch.load(path, map_location=device, weights_only=False)


def epoch_pass(model, loader, criterion, device, optimizer=None, scaler=None,
               accum_steps=1, grad_clip=1.0, amp=False):
    training = optimizer is not None
    model.train(training)
    total = {}
    count = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    # Correctly average gradient accumulation over the last, possibly short group.
    n_batches = len(loader)
    for ix, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        group_start = (ix // accum_steps) * accum_steps
        group_size = min(accum_steps, n_batches - group_start)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp):
                pred = model(x)
                losses = criterion(pred, y, x)
            if not torch.isfinite(losses['total']):
                raise FloatingPointError('Encountered NaN/Inf loss; try amp=false or smaller lr')
            if training:
                back_loss = losses['total'] / group_size
                if scaler is not None:
                    scaler.scale(back_loss).backward()
                else:
                    back_loss.backward()
                if (ix + 1) % accum_steps == 0 or (ix + 1) == n_batches:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        for key, value in losses.items():
            total[key] = total.get(key, 0.0) + float(value.detach())*x.size(0)
        count += x.size(0)
    if count == 0:
        raise ValueError('Empty dataset')
    return {key: value/count for key,value in total.items()}


def train(cfg: dict, resume: str | None = None):
    device = configure_device(str(cfg['hardware']['device']))
    seed_all(int(cfg['training']['seed']))
    folder = run_dir(cfg)
    ckdir = folder / 'checkpoints'
    last_path, best_path = ckdir / 'last_model.pth', ckdir / 'best_model.pth'
    if not resume and (last_path.exists() or best_path.exists()):
        raise FileExistsError(f'Run already exists: {folder}. Set a new experiment.run_name '
                              'or explicitly --resume a trusted checkpoint.')
    ckdir.mkdir(parents=True, exist_ok=True)
    if not resume:
        (folder/'config_used.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))

    train_loader = dataloader(cfg, 'train', train=True)
    val_loader = dataloader(cfg, 'val')
    model = HybridFNO3D.from_config(cfg['model']).to(device)
    criterion = ResidualLoss(**cfg['loss']).to(device)
    tc = cfg['training']
    optim = torch.optim.AdamW(model.parameters(), lr=float(tc['lr']),
                             weight_decay=float(tc['weight_decay']))
    epochs = int(tc['epochs'])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=epochs, eta_min=float(tc['min_lr']))
        if tc['scheduler'] == 'cosine' else None)
    if tc['scheduler'] not in ('cosine', 'none'):
        raise ValueError('training.scheduler must be cosine or none')
    amp = bool(tc['amp']) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=True) if amp else None
    start_epoch, best_val, best_epoch, stale = 1, float('inf'), 0, 0
    if resume:
        checkpoint = checkpoint_load(Path(resume), device)
        if cfg['model'] != checkpoint['config']['model']:
            raise ValueError('Model config changed from checkpoint. Resume with original architecture.')
        if float(cfg['data']['scale']) != float(checkpoint['config']['data']['scale']):
            raise ValueError('data.scale changed from checkpoint')
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None and checkpoint.get('scheduler') is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if scaler is not None and checkpoint.get('scaler'):
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val = checkpoint['best_val']
        best_epoch = checkpoint['best_epoch']
        stale = checkpoint['stale_epochs']
    if start_epoch > epochs:
        raise ValueError(f'Resumed epoch {start_epoch} exceeds training.epochs={epochs}')

    history_path = folder/'history.csv'
    new_csv = not history_path.exists() or not resume
    with history_path.open('a' if resume else 'w', newline='') as fp:
        fieldnames = ['epoch','train_total','train_pixel','train_gradient',
                      'val_total','val_pixel','val_gradient','lr','seconds']
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if new_csv:
            writer.writeheader()
        n_params = sum(p.numel() for p in model.parameters())
        print(f'Run: {folder}\nDevice: {device} | parameters: {n_params:,} | '
              f'train samples: {len(train_loader.dataset)} | val samples: {len(val_loader.dataset)}')
        for epoch in range(start_epoch, epochs+1):
            start = time.perf_counter()
            tr = epoch_pass(model, train_loader, criterion, device, optimizer=optim,
                            scaler=scaler, amp=amp, accum_steps=int(tc['grad_accum_steps']),
                            grad_clip=float(tc['grad_clip']))
            va = epoch_pass(model, val_loader, criterion, device, amp=amp)
            lr = optim.param_groups[0]['lr']
            better = va['total'] < best_val
            if better:
                best_val, best_epoch, stale = va['total'], epoch, 0
            else:
                stale += 1
            if scheduler is not None:
                scheduler.step()
            row = {
                'epoch': epoch, 'train_total': tr['total'], 'train_pixel': tr['pixel'],
                'train_gradient': tr.get('gradient', float('nan')),
                'val_total': va['total'], 'val_pixel': va['pixel'],
                'val_gradient': va.get('gradient', float('nan')),
                'lr': lr, 'seconds': round(time.perf_counter()-start, 2),
            }
            writer.writerow(row)
            fp.flush()
            checkpoint_save(last_path, model=model, optimizer=optim, scheduler=scheduler,
                            scaler=scaler, cfg=cfg, epoch=epoch, best_val=best_val,
                            best_epoch=best_epoch, stale_epochs=stale)
            if better:
                checkpoint_save(best_path, model=model, optimizer=optim, scheduler=scheduler,
                                scaler=scaler, cfg=cfg, epoch=epoch, best_val=best_val,
                                best_epoch=best_epoch, stale_epochs=stale)
            print(f'Epoch {epoch:03d}/{epochs}: train={tr["total"]:.6g} '
                  f'val={va["total"]:.6g} lr={lr:.3g} best={best_val:.6g} '
                  f'({row["seconds"]}s)', flush=True)
            if stale >= int(tc['early_stopping_patience']):
                print(f'Early stopping: no validation improvement for {stale} epochs')
                break
    print(f'Best validation loss {best_val:.6g} at epoch {best_epoch}; '
          f'checkpoint: {best_path}')
    return best_path


@torch.no_grad()
def evaluate(cfg: dict, split='test', checkpoint_path: str | None = None,
             save_predictions=False):
    device = configure_device(str(cfg['hardware']['device']))
    path = Path(checkpoint_path) if checkpoint_path else run_dir(cfg)/'checkpoints'/'best_model.pth'
    if not path.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {path}')
    state = checkpoint_load(path, device)
    saved_cfg = state['config']
    # Rebuild with architecture and shared scale from checkpoint, not mutable test config.
    cfg = copy.deepcopy(cfg)
    cfg['model'] = saved_cfg['model']
    cfg['data']['scale'] = saved_cfg['data']['scale']
    cfg['loss'] = saved_cfg['loss']
    loader = dataloader(cfg, split)
    model = HybridFNO3D.from_config(cfg['model']).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    criterion = ResidualLoss(**cfg['loss']).to(device)
    scale = float(cfg['data']['scale'])
    totals = {}
    rows = []
    pred_dir = run_dir(cfg)/'predictions'/split
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)
    cursor = 0
    for inp, correction in loader:
        inp = inp.to(device)
        correction = correction.to(device)
        output = model(inp)
        parts = criterion(output, correction, inp)
        prediction = (inp + output) * scale
        true_dns = (inp + correction) * scale
        baseline = inp * scale
        bsz = inp.size(0)
        err = (prediction-true_dns).reshape(bsz,-1)
        baseline_err = (baseline-true_dns).reshape(bsz,-1)
        corr_err = ((output-correction)*scale).reshape(bsz,-1)
        corr_target = (correction*scale).reshape(bsz,-1)
        true_flat = true_dns.reshape(bsz,-1)
        for i in range(bsz):
            ts = loader.dataset.pairs[cursor+i][0]
            row = dict(
                timestep=ts,
                reconstruction_mse=float(err[i].square().mean()),
                baseline_mse=float(baseline_err[i].square().mean()),
                reconstruction_rel_l2=float(torch.linalg.vector_norm(err[i]) /
                                             torch.linalg.vector_norm(true_flat[i]).clamp_min(1e-12)),
                correction_mse=float(corr_err[i].square().mean()),
                correction_rel_l2=float(torch.linalg.vector_norm(corr_err[i]) /
                                         torch.linalg.vector_norm(corr_target[i]).clamp_min(1e-12)),
            )
            rows.append(row)
            if save_predictions:
                arr = (output[i]*scale).detach().cpu().numpy().transpose(1,2,3,0)
                np.save(pred_dir/f'prediction_t{ts}.npy', arr.astype('float32'))
        cursor += bsz
        for k,v in parts.items():
            totals[k] = totals.get(k,0) + float(v)*bsz
    if not rows:
        raise ValueError(f'No samples to evaluate for split={split}')
    summary = {k:sum(row[k] for row in rows)/len(rows)
               for k in rows[0] if k!='timestep'}
    summary.update({f'loss_{k}':v/len(rows) for k,v in totals.items()})
    summary.update({'split':split, 'n_samples':len(rows), 'checkpoint':str(path),
                    'best_training_epoch':state['best_epoch']})
    # Evaluation outputs are always confined to this FNO project's run folder.
    result_dir = run_dir(cfg)/'metrics'
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir/f'{split}_summary.json').write_text(json.dumps(summary,indent=2))
    with (result_dir/f'{split}_per_timestep.csv').open('w',newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary,indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description='Standalone 3D residual FNO trainer')
    parser.add_argument('--config', type=Path, default=HERE/'config.yaml')
    parser.add_argument('-o','--override', action='append', default=[],
                        metavar='KEY=VALUE', help='Repeat for YAML settings; e.g. -o model.width=24')
    parser.add_argument('--resume', default=None, help='Trusted last_model.pth for resuming')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--split', default='test', choices=['train','val','test'])
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--save-predictions', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    if args.evaluate:
        evaluate(cfg, args.split, args.checkpoint, args.save_predictions)
    else:
        train(cfg, args.resume)


if __name__ == '__main__':
    main()
