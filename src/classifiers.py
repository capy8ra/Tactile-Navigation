import os
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances

from .features import extract_features


# ---------------------------------------------------------------------------
# CNN model
# ---------------------------------------------------------------------------

class TactileCNN(nn.Module):
    """
    1-D Convolutional Neural Network for tactile intent classification.

    Input shape:  (batch, input_features)  — flattened feature vector (550,).
    Output shape: (batch, n_classes)
    """

    def __init__(self, input_features=220, n_classes=5):
        super().__init__()
        self.input_features = input_features

        self.conv1 = nn.Conv1d(1, 8,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(8, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool1d(2)
        self.drop  = nn.Dropout(0.2)

        conv_output_size = input_features // 4   # two pooling layers
        self.fc1 = nn.Linear(32 * conv_output_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, n_classes)

    def forward(self, x):
        # x: (B, input_features) → (B, 1, input_features) for Conv1d
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)


# ---------------------------------------------------------------------------
# Unified classifier interface
# ---------------------------------------------------------------------------

class TactileClassifier:
    """
    Loads tactile data and provides training / evaluation routines for
    four baseline classifiers:  RBF-SVM · MLP · MDCM · CNN.

    All baselines share the same feature extraction pipeline (see
    src/features.py) so that comparisons are fair.
    """

    ACTIONS = ['forward', 'backward', 'left', 'right', 'static']

    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, data_dir='data/simulated'):
        """
        Load .npy files from *data_dir*.

        Expected file naming: {action}_{trial}.npy
        (e.g. forward_0.npy, left_3.npy, …)

        Returns:
            X: (n_samples, time_steps, 11, 5)
            y: (n_samples,) integer labels  [0..4]
        """
        X, y = [], []
        for label, action in enumerate(self.ACTIONS):
            trial = 0
            while True:
                path = os.path.join(data_dir, f'{action}_{trial}.npy')
                if not os.path.exists(path):
                    break
                X.append(np.load(path))
                y.append(label)
                trial += 1

        X = np.array(X)
        y = np.array(y, dtype=int)
        print(f"Loaded {len(X)} samples from '{data_dir}'  |  shape: {X.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        return X, y

    # ------------------------------------------------------------------
    # Baseline: RBF-SVM
    # ------------------------------------------------------------------

    def train_svm(self, X_train, y_train, X_test, y_test):
        """RBF-SVM trained on flattened feature vectors."""
        print("\n=== RBF-SVM ===")

        F_train = extract_features(X_train).reshape(len(X_train), -1)  # (n, 550)
        F_test  = extract_features(X_test).reshape(len(X_test),  -1)

        svm = SVC(kernel='rbf', gamma=1e-2, C=1.0, random_state=1)
        t0  = time.time()
        svm.fit(F_train, y_train)
        print(f"Training time: {time.time()-t0:.2f}s")

        y_pred   = svm.predict(F_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=self.ACTIONS))

        return svm, accuracy

    # ------------------------------------------------------------------
    # Baseline: MLP
    # ------------------------------------------------------------------

    def train_mlp(self, X_train, y_train, X_test, y_test):
        """MLP trained on standardized flattened feature vectors."""
        print("\n=== MLP ===")

        F_train = extract_features(X_train).reshape(len(X_train), -1)
        F_test  = extract_features(X_test).reshape(len(X_test),  -1)

        scaler  = StandardScaler()
        F_train = scaler.fit_transform(F_train)
        F_test  = scaler.transform(F_test)

        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 16, 4),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            batch_size='auto',
            learning_rate_init=1e-3,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )
        t0 = time.time()
        mlp.fit(F_train, y_train)
        print(f"Training time: {time.time()-t0:.2f}s  ({mlp.n_iter_} iterations)")

        y_pred   = mlp.predict(F_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=self.ACTIONS))

        return mlp, scaler, accuracy

    # ------------------------------------------------------------------
    # Baseline: MDCM (Minimum Distance to Covariance Mean)
    # ------------------------------------------------------------------

    def _build_covariance_matrices(self, X):
        """
        Build regularized covariance matrices for Riemannian classification.

        Each trial is represented by a (n_windows × n_window_features) matrix
        whose sample covariance (with regularization) is used as the SPD input.
        """
        cov_matrices = []
        window, stride = 10, 5

        for trial in X:
            # trial: (T, 11, 5)
            windows = []
            for i in range(0, trial.shape[0] - window + 1, stride):
                w = trial[i:i + window]             # (10, 11, 5)
                windows.append(np.concatenate([
                    w.mean(axis=0).ravel(),          # mean over window
                    w.std(axis=0).ravel(),           # std  over window
                ]))
            features = np.array(windows)            # (n_windows, 110)

            if features.shape[0] > features.shape[1]:
                cov = np.cov(features.T)
            else:  # more features than samples: add fixed regularization first
                cov = np.cov(features.T) + 0.1 * np.eye(features.shape[1])
            reg = 0.1 * np.trace(cov) / cov.shape[0]
            cov = (cov + cov.T) / 2 + reg * np.eye(cov.shape[0])
            cov_matrices.append(cov)

        return np.array(cov_matrices)

    def train_riemannian(self, X_train, y_train, X_test, y_test):
        """MDCM (Minimum Distance to Covariance Mean) on Riemannian manifold."""
        print("\n=== MDCM (Riemannian) ===")

        t0        = time.time()
        cov_train = self._build_covariance_matrices(X_train)
        cov_test  = self._build_covariance_matrices(X_test)
        print(f"Covariance prep: {time.time()-t0:.2f}s  |  shape: {cov_train.shape}")

        mdm = MDM(metric='riemann')
        t0  = time.time()
        mdm.fit(cov_train, y_train)
        print(f"Training time: {time.time()-t0:.2f}s")

        y_pred   = mdm.predict(cov_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=self.ACTIONS))

        return mdm, accuracy

    # ------------------------------------------------------------------
    # Baseline: CNN
    # ------------------------------------------------------------------

    def train_cnn(self, X_train, y_train, X_test, y_test, epochs=100):
        """1-D CNN trained on flattened feature vectors."""
        print("\n=== CNN ===")

        F_train = extract_features(X_train).reshape(len(X_train), -1)   # (n, 550)
        F_test  = extract_features(X_test).reshape(len(X_test),  -1)

        F_train = torch.FloatTensor(F_train).to(self.device)
        F_test  = torch.FloatTensor(F_test).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        y_test_t  = torch.LongTensor(y_test).to(self.device)

        loader = DataLoader(TensorDataset(F_train, y_train_t),
                            batch_size=4, shuffle=True)

        input_features = F_train.shape[1]
        model     = TactileCNN(input_features=input_features,
                               n_classes=len(self.ACTIONS)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)

        t0, losses = time.time(), []
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}  loss={losses[-1]:.4f}")
        print(f"Training time: {time.time()-t0:.2f}s")

        model.eval()
        with torch.no_grad():
            preds = model(F_test).argmax(dim=1)
        y_pred   = preds.cpu().numpy()
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=self.ACTIONS))

        return model, accuracy, losses

    # ------------------------------------------------------------------
    # Plotting utilities
    # ------------------------------------------------------------------

    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_path=None):
        """Save a confusion-matrix figure (raw counts + row-normalized %)."""
        cm      = confusion_matrix(y_true, y_pred)
        cm_pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        kw = dict(xticklabels=self.ACTIONS, yticklabels=self.ACTIONS)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    cbar_kws={'label': 'Count'}, ax=ax1, **kw)
        ax1.set_title(f'{model_name}  (acc={accuracy_score(y_true, y_pred):.3f})')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')

        annots = [[f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)'
                   for j in range(cm.shape[1])]
                  for i in range(cm.shape[0])]
        sns.heatmap(cm_pct, annot=annots, fmt='', cmap='Blues',
                    cbar_kws={'label': 'Row %'}, ax=ax2, **kw)
        ax2.set_title(f'{model_name}  (row-normalized)')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()

    def plot_accuracy_comparison(self, results: dict, save_path=None):
        """
        Bar chart comparing all classifiers.

        Args:
            results: {model_name: accuracy}  (floats in [0, 1])
        """
        names = list(results.keys())
        accs  = list(results.values())
        colors = ['#5B8DB8', '#2E8B57', '#E87A5D', '#F5C518', '#9B59B6']

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(names, accs, color=colors[:len(names)])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Accuracy')
        ax.set_title('Classifier Comparison on Simulated Data')
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    acc + 0.01, f'{acc:.3f}', ha='center', fontsize=9)
        ax.tick_params(axis='x', rotation=15)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
