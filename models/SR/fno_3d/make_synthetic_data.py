"""Generate tiny synthetic velocity cubes to smoke-test the training pipeline.

NOT scientifically realistic DNS. Never use these to claim SR accuracy.
"""
import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='synthetic_data')
    p.add_argument('--size', type=int, default=16)
    args = p.parse_args()
    if args.size < 8:
        p.error('--size must be at least 8')
    root = Path(args.output)
    rng = np.random.default_rng(42)
    z, y, x = np.meshgrid(*[np.linspace(0, 2*np.pi, args.size, endpoint=False)]*3,
                          indexing='ij')
    for split, count in [('train', 6), ('val', 2), ('test', 2)]:
        folder = root / split
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            phase = rng.uniform(0, 2*np.pi)
            u = np.sin(y+phase)*np.cos(z)
            v = np.sin(z+phase)*np.cos(x)
            w = np.sin(x+phase)*np.cos(y)
            inp = np.stack([u, v, w], axis=-1).astype('float32')
            target = (0.05 * np.stack([
                np.sin(2*x+phase), np.sin(2*y+phase), np.sin(2*z+phase)
            ], axis=-1)).astype('float32')
            np.save(folder / f'input_t{i:04d}.npy', inp)
            np.save(folder / f'target_t{i:04d}.npy', target)
    print(f'Wrote synthetic input/target pairs to {root.resolve()}')


if __name__ == '__main__':
    main()
