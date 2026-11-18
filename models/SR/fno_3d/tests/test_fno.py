"""Fast unit tests; run with python -m unittest discover -s tests -v."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import HybridFNO3D, SpectralConv3d
from dataset import VelocityCubePairs
from losses import ResidualLoss


class ModelTests(unittest.TestCase):
    def test_baseline_zero_init_and_shape(self):
        net = HybridFNO3D(width=4, modes=[3,3,3], depth=1)
        x = torch.randn(1,3,8,8,8)
        correction = net(x)
        self.assertEqual(correction.shape, x.shape)
        self.assertTrue(torch.all(correction == 0))

    def test_fft_gradients(self):
        net = HybridFNO3D(width=4, modes=[3,3,3], depth=1,
                          zero_init_head=False)
        x = torch.randn(1,3,8,8,8)
        y = net(x)
        y.square().mean().backward()
        grad = net.blocks[0].spectral.weights[0].grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())

    def test_mode_limit(self):
        net = SpectralConv3d(width=4, modes=[9,3,3])
        with self.assertRaisesRegex(ValueError, 'Fourier modes'):
            net(torch.randn(1,4,8,8,8))

    def test_loss(self):
        net = ResidualLoss(pixel_weight=1,gradient_weight=.1,spectral_weight=.02)
        inp = torch.randn(1,3,8,8,8)
        target = torch.randn_like(inp)*0.01
        pred = torch.zeros_like(inp, requires_grad=True)
        loss = net(pred,target,inp)['total']
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_data_pairs(self):
        with tempfile.TemporaryDirectory() as base:
            d=Path(base)/'train'; d.mkdir()
            a=np.ones((8,8,8,3),dtype='float32')
            np.save(d/'input_t0010.npy',a)
            np.save(d/'target_t0010.npy',a*0.1)
            ds=VelocityCubePairs(base,'train',scale=2)
            x,y=ds[0]
            self.assertEqual(x.shape,(3,8,8,8))
            self.assertAlmostEqual(float(x[0,0,0,0]),.5)
            self.assertAlmostEqual(float(y[0,0,0,0]),.05,places=6)


if __name__ == '__main__':
    unittest.main()
