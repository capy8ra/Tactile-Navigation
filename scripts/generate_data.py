"""
Generate the simulated tactile dataset used in the paper.

Each of the five intents is assigned a distinct spatial base pattern on the
11 × 5 tactile grid.  For every trial the pattern is:
  1. Rolled by a random row offset  (simulates rotational shift in grasp)
  2. Corrupted with Gaussian spatial noise
  3. Expanded into a time-series with realistic temporal dynamics
  4. Corrupted with mild temporal noise

Output: data/simulated/{action}_{trial}.npy   shape (91, 11, 5)

Usage:
    python scripts/generate_data.py [--output_dir data/simulated] [--n_trials 40]
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Base spatial patterns
# ---------------------------------------------------------------------------

def create_base_patterns():
    """
    Five distinct spatial patterns — one per navigation intent.

    Sensor grid: 11 rows (cylinder height) × 5 cols (cylinder width).
    Values in [0, 1]; rows wrap cylindrically (row 0 ↔ row 10).
    """
    patterns = {}

    # Forward: X / cross pattern
    fwd = np.zeros((11, 5))
    for i in range(5):
        if 3 + i < 11:
            fwd[3 + i, i]     = 1.0   # diagonal ↘
            fwd[3 + i, 4 - i] = 1.0   # diagonal ↙
    patterns['forward'] = fwd

    # Backward: concentrated lower-region activation
    bwd = np.zeros((11, 5))
    bwd[7:10, 1:4] = 1.0
    bwd[9:11, 2]   = 0.8
    bwd[5:7,  2]   = 0.6
    bwd[8,    :]   = 0.9
    patterns['backward'] = bwd

    # Left: left-side lateral activation
    left = np.zeros((11, 5))
    left[3:8, 0:2] = 1.0
    left[4:7, 2:4] = 0.7
    left[5,   :]   = 0.8
    left[2:9, 0]   = 0.9
    patterns['left'] = left

    # Right: right-side lateral activation (mirror of left)
    right = np.zeros((11, 5))
    right[3:8, 3:5] = 1.0
    right[4:7, 1:3] = 0.7
    right[5,   :]   = 0.8
    right[2:9, 4]   = 0.9
    patterns['right'] = right

    # Static: 3 × 3 central square
    static = np.zeros((11, 5))
    static[4:7, 1:4] = 1.0
    patterns['static'] = static

    return patterns


# ---------------------------------------------------------------------------
# Temporal dynamics
# ---------------------------------------------------------------------------

def add_temporal_dynamics(base_pattern, time_steps=91, action='forward'):
    """
    Expand a static spatial pattern into a (time_steps, 11, 5) time-series.

    Static intent: gradual build-up then sustained hold.
    Dynamic intents: rise → peak (with sinusoidal variation) → decay.
    """
    seq = np.zeros((time_steps, 11, 5))

    if action == 'static':
        for t in range(time_steps):
            intensity = (t / (time_steps // 3)
                         if t < time_steps // 3
                         else 1.0 + 0.1 * np.sin(t * 0.2))
            seq[t] = base_pattern * intensity
    else:
        t_rise = time_steps // 4
        t_peak = 3 * time_steps // 4
        for t in range(time_steps):
            if t < t_rise:
                intensity = 0.2 + 0.6 * (t / t_rise)
            elif t < t_peak:
                intensity = 0.8 + 0.4 * np.sin((t - t_rise) * 0.3)
            else:
                progress  = (t - t_peak) / (time_steps - t_peak)
                intensity = 0.8 * (1 - progress) + 0.2
            seq[t] = base_pattern * intensity * (1.0 + 0.1 * np.random.randn())

    return seq


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(output_dir='data/simulated', n_trials=40, noise=0.05,
                     time_steps=91, seed=42):
    """
    Generate and save the simulated tactile dataset.

    Args:
        output_dir: directory to write .npy files
        n_trials:   samples per intent class
        noise:      spatial noise standard deviation
        time_steps: temporal length of each sample
        seed:       random seed for reproducibility
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    base_patterns = create_base_patterns()
    samples = {action: [] for action in base_patterns}

    for action, base in base_patterns.items():
        for trial in range(n_trials):
            # Random cylindrical shift (0–10 rows)
            shift = np.random.randint(0, 11)
            shifted = np.roll(base, shift, axis=0)

            # Spatial noise
            noisy = np.maximum(0, shifted + noise * np.random.randn(11, 5))

            # Temporal expansion + temporal noise
            seq = add_temporal_dynamics(noisy, time_steps, action)
            seq = np.maximum(0, seq + (noise * 0.5) * np.random.randn(time_steps, 11, 5))

            path = os.path.join(output_dir, f'{action}_{trial}.npy')
            np.save(path, seq.astype(np.float32))
            samples[action].append(seq)

    n_total = len(base_patterns) * n_trials
    print(f"Saved {n_total} samples ({n_trials} per class) → '{output_dir}'")
    return base_patterns, samples


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_patterns(base_patterns, samples, save_path=None):
    """
    Plot the five base patterns and two example time steps for each.
    Row 0: base patterns
    Row 1: early time step (t=20)
    Row 2: final time step (t=90)
    """
    actions = list(base_patterns.keys())
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))

    for col, action in enumerate(actions):
        # Base pattern
        axes[0, col].imshow(base_patterns[action], cmap='hot', interpolation='nearest')
        axes[0, col].set_title(f'{action.capitalize()} — Base')

        sample = samples[action][0]
        axes[1, col].imshow(sample[20],  cmap='hot', interpolation='nearest')
        axes[1, col].set_title(f'{action.capitalize()} — t=20')

        axes[2, col].imshow(sample[-1],  cmap='hot', interpolation='nearest')
        axes[2, col].set_title(f'{action.capitalize()} — t=90')

    for ax in axes.ravel():
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')

    plt.suptitle('Simulated Tactile Patterns', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate simulated tactile dataset.')
    parser.add_argument('--output_dir', default='data/simulated',
                        help='Directory to write .npy files (default: data/simulated)')
    parser.add_argument('--n_trials', type=int, default=40,
                        help='Number of trials per intent class (default: 40)')
    parser.add_argument('--noise', type=float, default=0.05,
                        help='Spatial noise level (default: 0.05)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--visualize', action='store_true',
                        help='Plot and save pattern visualizations')
    args = parser.parse_args()

    base_patterns, samples = generate_dataset(
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        noise=args.noise,
        seed=args.seed,
    )

    if args.visualize:
        import os
        vis_path = os.path.join(args.output_dir, '..', 'simulated_patterns.png')
        visualize_patterns(base_patterns, samples, save_path=vis_path)
