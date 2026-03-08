import numpy as np


def extract_features(X):
    """
    Extract spatiotemporal features from tactile sensor data.

    Collapses the time-series dimension into 4 feature maps that preserve
    the sensor's cylindrical spatial topology (11 rows × 5 columns).

    Features (stacked along the last axis), as described in the paper:
        0  mean_activation  - average pressure distribution over time
        1  max_activation   - peak pressure over time
        2  std_activation   - temporal variability of pressure
        3  spatial_gradient - spatial edge magnitude of the mean map

    Args:
        X: array-like of shape (n_samples, time_steps, 11, 5)

    Returns:
        features: numpy array of shape (n_samples, 11, 5, 4)
    """
    X = np.asarray(X, dtype=np.float32)   # (n, T, 11, 5)

    mean_act = np.mean(X, axis=1)          # (n, 11, 5)
    max_act  = np.max(X,  axis=1)          # (n, 11, 5)
    std_act  = np.std(X,  axis=1)          # (n, 11, 5)

    spatial_grad_h   = np.gradient(mean_act, axis=2)   # along width  (cols)
    spatial_grad_v   = np.gradient(mean_act, axis=1)   # along height (rows)
    spatial_gradient = np.sqrt(spatial_grad_h**2 + spatial_grad_v**2)  # (n, 11, 5)

    # Stack → (n_samples, 11, 5, 4)
    return np.stack([mean_act, max_act, std_act, spatial_gradient], axis=-1)
