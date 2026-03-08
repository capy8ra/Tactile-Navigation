"""
Visualize a single tactile data trial.

Loads one .npy file (action + trial index) and shows:
  - A heatmap animation across all time steps
  - A grid of evenly-spaced time-step snapshots

Usage:
    python scripts/visualize_data.py --action forward --trial 0
    python scripts/visualize_data.py --action left --trial 3 --data_dir data/simulated
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

ACTIONS = ['forward', 'backward', 'left', 'right', 'static']


def load_trial(data_dir, action, trial):
    path = os.path.join(data_dir, f'{action}_{trial}.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    data = np.load(path)          # (T, 11, 5)
    print(f"Loaded '{path}'  shape={data.shape}  "
          f"min={data.min():.3f}  max={data.max():.3f}")
    return data


def plot_snapshots(data, action, trial, n_cols=6):
    """Grid of evenly-spaced time-step snapshots."""
    T = data.shape[0]
    indices = np.linspace(0, T - 1, n_cols * 2, dtype=int)
    n_rows = (len(indices) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 3))
    fig.suptitle(f'Action: {action.capitalize()}  |  Trial: {trial}', fontsize=13)
    axes = axes.ravel()

    vmin, vmax = data.min(), data.max()
    for ax, t in zip(axes, indices):
        im = ax.imshow(data[t], cmap='hot', vmin=vmin, vmax=vmax,
                       interpolation='nearest', aspect='auto')
        ax.set_title(f't = {t}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for ax in axes[len(indices):]:
        ax.axis('off')

    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label='Pressure (a.u.)')
    plt.tight_layout()
    plt.show()


def animate_trial(data, action, trial, interval=60):
    """Animated heatmap scrolling through all time steps."""
    T = data.shape[0]
    vmin, vmax = data.min(), data.max()

    fig, ax = plt.subplots(figsize=(4, 6))
    im = ax.imshow(data[0], cmap='hot', vmin=vmin, vmax=vmax,
                   interpolation='nearest', aspect='auto')
    ax.set_xlabel('Width (cols)')
    ax.set_ylabel('Height (rows)')
    title = ax.set_title(f'{action.capitalize()}  |  trial {trial}  |  t = 0')
    fig.colorbar(im, ax=ax, label='Pressure (a.u.)')

    def update(t):
        im.set_data(data[t])
        title.set_text(f'{action.capitalize()}  |  trial {trial}  |  t = {t}')
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=T,
                                  interval=interval, blit=True)
    plt.tight_layout()
    plt.show()
    return ani


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize one tactile data trial.')
    parser.add_argument('--action',   required=True, choices=ACTIONS,
                        help='Intent class to visualize')
    parser.add_argument('--trial',    type=int, default=0,
                        help='Trial index (default: 0)')
    parser.add_argument('--data_dir', default='data/final_data',
                        help='Data directory (default: data/simulated)')
    parser.add_argument('--animate',  action='store_true',
                        help='Also show animated time-series heatmap')
    args = parser.parse_args()

    data = load_trial(args.data_dir, args.action, args.trial)
    plot_snapshots(data, args.action, args.trial)
    if args.animate:
        animate_trial(data, args.action, args.trial)
