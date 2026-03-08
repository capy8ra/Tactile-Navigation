import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from .features import extract_features


class CylinderKernelSVM:
    """
    Cylindrical Kernel Support Vector Machine (CK-SVM).

    Classifies tactile intent patterns on a cylindrical sensor (11 rows × 5 cols)
    by replacing the standard RBF Euclidean distance with a Cylindrical Distance
    that finds the minimum-cost rotational alignment between two patterns:

        d_C(x1, x2) = min_{s=0..k-1} ( ||x1 - τ_s(x2)||²_F + Λ(s) )

        Λ(s) = exp( min(s, k-s) / δ ) - 1        (exponential shift penalty)

    This makes the classifier robust to natural rotational shifts in the user's
    grasp while still penalizing large misalignments.

    Reference:
        Peng et al., "Tactile-Based Human Intent Recognition for Robot Assistive
        Navigation", ICRA 2026.
    """

    N_ROWS = 11   # Sensor height (circumference of the cylinder)

    def __init__(self, gamma=1e-4, C=1.0, shift_penalty=5):
        """
        Args:
            gamma:         RBF kernel length-scale (controls kernel bandwidth).
            C:             SVM regularization parameter.
            shift_penalty: Exponential penalty coefficient δ. Higher values
                           enforce stricter penalties for large rotational shifts.
        """
        self.gamma         = gamma
        self.C             = C
        self.shift_penalty = shift_penalty
        self.svm           = None
        self.train_patterns = None  # (n_train, 11, 5, 10) normalized

    # ------------------------------------------------------------------
    # Internal kernel helpers
    # ------------------------------------------------------------------

    def _min_cyl_dist(self, shift):
        """Shortest rotational distance on the cylinder for a given row shift."""
        return min(shift, self.N_ROWS - shift)

    def _shift_penalties(self):
        """Pre-compute the exponential shift penalty for all 11 shifts."""
        dists = np.array([self._min_cyl_dist(s) for s in range(self.N_ROWS)])
        return np.exp(self.shift_penalty * dists) - 1   # shape (11,)

    @staticmethod
    def _normalize(patterns):
        """Per-sample, per-feature channel normalization. Input: (n, 11, 5, c)."""
        mean = patterns.mean(axis=(1, 2), keepdims=True)
        std  = patterns.std(axis=(1, 2), keepdims=True)
        return (patterns - mean) / (std + 1e-8)

    def _batch_kernel(self, batch, p2, penalties):
        """
        Cylindrical kernel for a batch of patterns vs one reference pattern.

        Args:
            batch:     (B, 11, 5, c) — normalized
            p2:        (11, 5, c)    — normalized
            penalties: (11,)         — pre-computed shift penalties

        Returns:
            kernel values: (B,)
        """
        # All row-shifted versions of p2: (11, 11, 5, c)
        shifts = np.stack([np.roll(p2, s, axis=0) for s in range(self.N_ROWS)])

        # Squared Frobenius distances for each (batch sample, shift): (B, 11)
        diff      = batch[:, np.newaxis] - shifts[np.newaxis]   # (B, 11, 11, 5, c)
        sq_dist   = np.sum(diff**2, axis=(2, 3, 4))             # (B, 11)

        # Add shift penalty and take the minimum over shifts
        penalized = sq_dist + penalties[np.newaxis]             # (B, 11)
        min_dist  = np.min(penalized, axis=1)                   # (B,)

        return np.exp(-self.gamma * min_dist)

    def _kernel_matrix(self, p1, p2=None):
        """
        Compute the (n1 × n2) cylindrical kernel matrix.

        Args:
            p1: (n1, 11, 5, c) — raw (un-normalized) feature patterns
            p2: (n2, 11, 5, c) or None.  If None, computes the symmetric
                training kernel (p1 vs p1).

        Returns:
            K: (n1, n2) kernel matrix
        """
        symmetric = p2 is None
        if symmetric:
            p2 = p1

        p1_n = self._normalize(p1)
        p2_n = self._normalize(p2)
        penalties = self._shift_penalties()

        n1, n2 = len(p1_n), len(p2_n)
        K = np.zeros((n1, n2))

        batch_size = min(50, n1)
        n_batches  = (n1 + batch_size - 1) // batch_size
        for b_idx in range(n_batches):
            i_start = b_idx * batch_size
            i_end   = min(i_start + batch_size, n1)
            batch   = p1_n[i_start:i_end]

            pct = (b_idx + 1) / n_batches * 100
            if b_idx % max(1, n_batches // 4) == 0:
                print(f"  Kernel matrix: {pct:.0f}% ({b_idx+1}/{n_batches} batches)")

            for j in range(n2):
                K[i_start:i_end, j] = self._batch_kernel(batch, p2_n[j], penalties)

        if symmetric:
            K = (K + K.T) / 2
            np.fill_diagonal(K, 1.0)

        return K

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit CK-SVM on training data.

        Args:
            X: (n_samples, time_steps, 11, 5) tactile data
            y: (n_samples,) integer class labels
        """
        print(f"[CK-SVM] Extracting features from {len(X)} training samples...")
        self.train_patterns = extract_features(X)   # (n, 11, 5, 10)

        print(f"[CK-SVM] Computing {len(X)}×{len(X)} training kernel matrix...")
        K_train = self._kernel_matrix(self.train_patterns)

        print("[CK-SVM] Fitting SVM with precomputed kernel...")
        self.svm = SVC(kernel='precomputed', C=self.C)
        self.svm.fit(K_train, y)
        print("[CK-SVM] Training complete.")
        return self

    def predict(self, X):
        """
        Predict intent labels for test data.

        Args:
            X: (n_samples, time_steps, 11, 5) tactile data

        Returns:
            predictions: (n_samples,) integer class labels
        """
        test_patterns = extract_features(np.asarray(X))
        print(f"[CK-SVM] Computing {len(X)}×{len(self.train_patterns)} test kernel matrix...")
        K_test = self._kernel_matrix(test_patterns, self.train_patterns)
        return self.svm.predict(K_test)

    def score(self, X, y):
        """Return classification accuracy on (X, y)."""
        return accuracy_score(y, self.predict(X))
