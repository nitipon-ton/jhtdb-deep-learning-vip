from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_plane(
    field: np.ndarray,
    title: str = "ω_z",
    save_path: Optional[Path] = None,
    cmap: str = "plasma",
    figsize: Tuple[int, int] = (6, 5),
    show_stats: bool = True,
) -> None:
    """
    Plot a single 2-D scalar field with statistics.

    Args:
        field: 2D array to plot
        title: Plot title
        save_path: Optional path to save figure
        cmap: Colormap name
        figsize: Figure size (width, height)
        show_stats: If True, display statistics on plot
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(field.T, origin='lower', cmap=cmap, aspect='auto')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add statistics as text
    if show_stats:
        stats_text = (
            f"Shape: {field.shape}\n"
            f"Min: {field.min():.3e}\n"
            f"Max: {field.max():.3e}\n"
            f"Mean: {field.mean():.3e}\n"
            f"Std: {field.std():.3e}"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('z', fontsize=10)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_multi_component(
    data: np.ndarray,
    components: List[str],
    title: str = "Velocity Components",
    save_path: Optional[Path] = None,
    cmap: str = "viridis",
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Plot multiple components side by side.

    Args:
        data: Array of shape (n_components, height, width)
        components: List of component names
        title: Overall title
        save_path: Optional path to save figure
        cmap: Colormap name
        figsize: Figure size (auto-computed if None)
    """
    n_components = data.shape[0]

    if figsize is None:
        figsize = (5 * n_components, 4)

    fig, axes = plt.subplots(1, n_components, figsize=figsize)

    if n_components == 1:
        axes = [axes]

    for idx, (ax, comp_name) in enumerate(zip(axes, components)):
        field = data[idx]

        im = ax.imshow(field.T, origin='lower', cmap=cmap, aspect='auto')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(f"{comp_name}", fontsize=11, fontweight='bold')
        ax.set_xlabel('x', fontsize=9)
        ax.set_ylabel('z', fontsize=9)

        # Add statistics
        stats_text = f"min={field.min():.2e}\nmax={field.max():.2e}"
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_lr_hr_comparison(
    lr: np.ndarray,
    hr: np.ndarray,
    title: str = "LR vs HR Comparison",
    save_path: Optional[Path] = None,
    cmap: str = "viridis",
) -> None:
    """
    Plot LR and HR side by side for comparison.

    Args:
        lr: Low-resolution field (H_lr, W_lr)
        hr: High-resolution field (H_hr, W_hr)
        title: Plot title
        save_path: Optional path to save figure
        cmap: Colormap name
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Determine shared colorbar limits
    vmin = min(lr.min(), hr.min())
    vmax = max(lr.max(), hr.max())

    # LR plot
    im0 = axes[0].imshow(lr.T, origin='lower', cmap=cmap,
                         vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title(f"Low Resolution\n{lr.shape[0]}×{lr.shape[1]}",
                      fontsize=11, fontweight='bold')
    axes[0].set_xlabel('x', fontsize=9)
    axes[0].set_ylabel('z', fontsize=9)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Add statistics
    lr_stats = (
        f"Mean: {lr.mean():.3e}\n"
        f"Std: {lr.std():.3e}"
    )
    axes[0].text(
        0.02, 0.98, lr_stats,
        transform=axes[0].transAxes,
        fontsize=8,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # HR plot
    im1 = axes[1].imshow(hr.T, origin='lower', cmap=cmap,
                         vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title(f"High Resolution\n{hr.shape[0]}×{hr.shape[1]}",
                      fontsize=11, fontweight='bold')
    axes[1].set_xlabel('x', fontsize=9)
    axes[1].set_ylabel('z', fontsize=9)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Add statistics
    hr_stats = (
        f"Mean: {hr.mean():.3e}\n"
        f"Std: {hr.std():.3e}"
    )
    axes[1].text(
        0.02, 0.98, hr_stats,
        transform=axes[1].transAxes,
        fontsize=8,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_patch_samples(
    patches: np.ndarray,
    n_samples: int = 6,
    title: str = "Patch Samples",
    save_path: Optional[Path] = None,
    cmap: str = "viridis",
) -> None:
    """
    Plot a grid of random patch samples.

    Args:
        patches: Array of shape (N, H, W) or (N, 1, H, W)
        n_samples: Number of samples to display
        title: Plot title
        save_path: Optional path to save figure
        cmap: Colormap name
    """
    # Handle channel dimension
    if patches.ndim == 4:
        patches = patches[:, 0, :, :]

    n_samples = min(n_samples, patches.shape[0])

    # Determine grid size
    n_cols = min(3, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    if n_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Select random samples
    indices = np.random.choice(patches.shape[0], n_samples, replace=False)

    for idx, ax in enumerate(axes[:n_samples]):
        patch = patches[indices[idx]]

        im = ax.imshow(patch.T, origin='lower', cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(f"Sample {indices[idx]}", fontsize=10)
        ax.set_xlabel('x', fontsize=8)
        ax.set_ylabel('z', fontsize=8)

        # Add statistics
        stats_text = f"μ={patch.mean():.2e}\nσ={patch.std():.2e}"
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_lr_hr_patches(
    lr_patches: np.ndarray,
    hr_patches: np.ndarray,
    n_samples: int = 3,
    title: str = "LR/HR Patch Pairs",
    save_path: Optional[Path] = None,
    cmap: str = "viridis",
) -> None:
    """
    Plot LR and HR patch pairs side by side.

    Args:
        lr_patches: LR patches of shape (N, H_lr, W_lr) or (N, 1, H_lr, W_lr)
        hr_patches: HR patches of shape (N, H_hr, W_hr) or (N, 1, H_hr, W_hr)
        n_samples: Number of sample pairs to display
        title: Plot title
        save_path: Optional path to save figure
        cmap: Colormap name
    """
    # Handle channel dimension
    if lr_patches.ndim == 4:
        lr_patches = lr_patches[:, 0, :, :]
    if hr_patches.ndim == 4:
        hr_patches = hr_patches[:, 0, :, :]

    n_samples = min(n_samples, lr_patches.shape[0], hr_patches.shape[0])

    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    # Select random samples
    indices = np.random.choice(lr_patches.shape[0], n_samples, replace=False)

    for row, idx in enumerate(indices):
        lr_patch = lr_patches[idx]
        hr_patch = hr_patches[idx]

        # Shared colorbar limits
        vmin = min(lr_patch.min(), hr_patch.min())
        vmax = max(lr_patch.max(), hr_patch.max())

        # LR
        im0 = axes[row, 0].imshow(lr_patch.T, origin='lower', cmap=cmap,
                                   vmin=vmin, vmax=vmax, aspect='auto')
        axes[row, 0].set_title(f"LR Patch {idx} ({lr_patch.shape[0]}×{lr_patch.shape[1]})",
                               fontsize=10)
        axes[row, 0].set_xlabel('x', fontsize=8)
        axes[row, 0].set_ylabel('z', fontsize=8)
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        # HR
        im1 = axes[row, 1].imshow(hr_patch.T, origin='lower', cmap=cmap,
                                   vmin=vmin, vmax=vmax, aspect='auto')
        axes[row, 1].set_title(f"HR Patch {idx} ({hr_patch.shape[0]}×{hr_patch.shape[1]})",
                               fontsize=10)
        axes[row, 1].set_xlabel('x', fontsize=8)
        axes[row, 1].set_ylabel('z', fontsize=8)
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_data_histogram(
    data: np.ndarray,
    title: str = "Data Distribution",
    save_path: Optional[Path] = None,
    bins: int = 50,
) -> None:
    """
    Plot histogram of data values.

    Args:
        data: Data array
        title: Plot title
        save_path: Optional path to save figure
        bins: Number of histogram bins
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    data_flat = data.ravel()

    ax.hist(data_flat, bins=bins, density=True, alpha=0.7,
            edgecolor='black', color='steelblue')
    ax.axvline(data_flat.mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {data_flat.mean():.3e}')
    ax.axvline(data_flat.mean() + data_flat.std(), color='orange',
               linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'±1σ: {data_flat.std():.3e}')
    ax.axvline(data_flat.mean() - data_flat.std(), color='orange',
               linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Value', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Add statistics text box
    stats_text = (
        f"N = {data.size}\n"
        f"Min = {data_flat.min():.3e}\n"
        f"Max = {data_flat.max():.3e}\n"
        f"Mean = {data_flat.mean():.3e}\n"
        f"Std = {data_flat.std():.3e}\n"
        f"Median = {np.median(data_flat):.3e}"
    )
    ax.text(
        0.98, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
    )

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()

