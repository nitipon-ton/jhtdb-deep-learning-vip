"""Losses for full-resolution 3D residual velocity targets."""
import torch
from torch import nn
from torch.nn import functional as F


def gradient_mse(pred: torch.Tensor, target: torch.Tensor, periodic: bool = True):
    loss = pred.new_zeros(())
    for axis in (-3, -2, -1):
        if periodic:
            delta = torch.roll(pred - target, -1, dims=axis) - (pred - target)
        else:
            diff = pred - target
            delta = diff.diff(dim=axis)
        loss = loss + delta.square().mean()
    return loss


def spectral_magnitude_mse(pred: torch.Tensor, target: torch.Tensor):
    # Not a shell-averaged turbulent energy spectrum. It is a phase-insensitive
    # regularizer on Fourier magnitudes and costs two additional 3D FFTs.
    with torch.autocast(device_type=pred.device.type, enabled=False):
        pf = torch.fft.rfftn(pred.float(), dim=(-3, -2, -1), norm='ortho')
        tf = torch.fft.rfftn(target.float(), dim=(-3, -2, -1), norm='ortho')
        # normalized squared error (stable across resolutions and amplitudes)
        return (pf.abs() - tf.abs()).square().mean() / tf.abs().square().mean().clamp_min(1e-8)


def divergence(velocity: torch.Tensor, spacing=(1.0, 1.0, 1.0)):
    """Discrete forward-difference divergence; channels (u,v,w), axes (Z,Y,X).
    spacing order: [dz,dy,dx]. Only meaningful with periodic velocity cubes.
    """
    dz, dy, dx = [float(s) for s in spacing]
    if min(dz, dy, dx) <= 0:
        raise ValueError('spacing must contain positive dz,dy,dx')
    u, v, w = velocity[:, 0], velocity[:, 1], velocity[:, 2]
    return ((torch.roll(u, -1, -1) - u)/dx
            + (torch.roll(v, -1, -2) - v)/dy
            + (torch.roll(w, -1, -3) - w)/dz)


class ResidualLoss(nn.Module):
    def __init__(self, pixel_weight=1.0, gradient_weight=0.1,
                 spectral_weight=0.0, divergence_weight=0.0,
                 periodic=True, spacing=(1.0, 1.0, 1.0)):
        super().__init__()
        self.weights = {
            'pixel': float(pixel_weight), 'gradient': float(gradient_weight),
            'spectral': float(spectral_weight), 'divergence': float(divergence_weight)
        }
        if any(v < 0 for v in self.weights.values()) or sum(self.weights.values()) == 0:
            raise ValueError('Loss weights must be >= 0, with one nonzero')
        self.periodic = bool(periodic)
        self.spacing = spacing
        if self.weights['divergence'] and not periodic:
            raise ValueError('This divergence loss assumes periodic boundaries; set weight=0 otherwise')

    def forward(self, correction: torch.Tensor, target: torch.Tensor,
                input_field: torch.Tensor):
        correction, target, input_field = (x.float() for x in (correction, target, input_field))
        metrics = {'pixel': F.mse_loss(correction, target)}
        if self.weights['gradient']:
            metrics['gradient'] = gradient_mse(correction, target, self.periodic)
        if self.weights['spectral']:
            # Spectral error on reconstructed full fields, not residuals.
            metrics['spectral'] = spectral_magnitude_mse(input_field + correction, input_field + target)
        if self.weights['divergence']:
            metrics['divergence'] = divergence(input_field + correction, self.spacing).square().mean()
        metrics['total'] = sum(self.weights[k]*v for k,v in metrics.items())
        return metrics
