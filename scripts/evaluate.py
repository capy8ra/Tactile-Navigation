"""
Train and evaluate all classifiers on the simulated tactile dataset.

Reproduces the "Simulated Dataset" row of Table I in the paper:
    CK-SVM · RBF-SVM · MLP · MDCM · CNN

Results (accuracy ± std) are estimated over multiple random train/test splits
and printed in a summary table. Confusion matrices are saved to results/.

Usage:
    # Quick single run
    python scripts/evaluate.py

    # Multiple random splits (as reported in the paper)
    python scripts/evaluate.py --n_runs 10 --data_dir data/simulated
"""

import argparse
import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

# Allow running from the repo root: python scripts/evaluate.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ck_svm import CylinderKernelSVM
from src.classifiers import TactileClassifier


# ---------------------------------------------------------------------------
# Single evaluation run
# ---------------------------------------------------------------------------

def run_once(data_dir, test_size, random_state, results_dir, gamma, C):
    """Load data, split, train all classifiers, return accuracy dict."""
    tc = TactileClassifier(results_dir=results_dir)
    X, y = tc.load_data(data_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}  "
          f"(seed={random_state})\n{'='*60}")

    accs = {}

    # CK-SVM (proposed method)
    ck = CylinderKernelSVM(gamma=gamma, C=C)
    t0 = time.time()
    ck.fit(X_train, y_train)
    y_pred = ck.predict(X_test)
    accs['CK-SVM'] = (y_pred == y_test).mean()
    print(f"[CK-SVM]   acc={accs['CK-SVM']:.4f}  ({time.time()-t0:.1f}s)")

    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred, target_names=tc.ACTIONS))


    # Baselines
    _, svm_acc = tc.train_svm(X_train, y_train, X_test, y_test)
    accs['RBF-SVM'] = svm_acc

    _, _, mlp_acc = tc.train_mlp(X_train, y_train, X_test, y_test)
    accs['MLP'] = mlp_acc

    _, mdcm_acc = tc.train_riemannian(X_train, y_train, X_test, y_test)
    accs['MDCM'] = mdcm_acc

    _, cnn_acc, _ = tc.train_cnn(X_train, y_train, X_test, y_test)
    accs['CNN'] = cnn_acc

    return accs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate tactile intent classifiers on simulated data.'
    )
    parser.add_argument('--data_dir',    default='data/final_data',
                        help='Directory containing .npy data files')
    parser.add_argument('--results_dir', default='results',
                        help='Directory to save confusion matrices and plots')
    parser.add_argument('--test_size',   type=float, default=0.2,
                        help='Fraction of data used for testing (default: 0.2)')
    parser.add_argument('--n_runs',      type=int,   default=10,
                        help='Number of random train/test splits to average over')
    parser.add_argument('--gamma',       type=float, default=1e-2,
                        help='CK-SVM gamma parameter (default: 1e-4)')
    parser.add_argument('--C',           type=float, default=1.0,
                        help='CK-SVM C parameter (default: 1.0)')
    args = parser.parse_args()

    all_accs = {k: [] for k in ['RBF-SVM', 'MLP', 'MDCM', 'CNN', 'CK-SVM']}

    for run in range(args.n_runs):
        seed = np.random.randint(10000)
        print(f"\n{'#'*60}")
        print(f"  Run {run+1}/{args.n_runs}  (seed={seed})")
        print(f"{'#'*60}")

        accs = run_once(
            data_dir     = args.data_dir,
            test_size    = args.test_size,
            random_state = seed,
            results_dir  = args.results_dir,
            gamma        = args.gamma,
            C            = args.C,
        )
        for k, v in accs.items():
            all_accs[k].append(v)

    # Summary table
    print(f"\n{'='*60}")
    print(f"  RESULTS  ({args.n_runs} run{'s' if args.n_runs > 1 else ''})")
    print(f"{'='*60}")
    header = f"{'Classifier':<12}  {'Mean':>7}  {'Std':>7}"
    print(header)
    print('-' * len(header))
    for name, vals in all_accs.items():
        mu  = np.mean(vals)
        std = np.std(vals)
        print(f"{name:<12}  {mu:>7.4f}  {std:>7.4f}")
    print(f"{'='*60}")

    # Save accuracy comparison bar chart
    tc = TactileClassifier(results_dir=args.results_dir)
    tc.plot_accuracy_comparison(
        {k: np.mean(v) for k, v in all_accs.items()},
        save_path=os.path.join(args.results_dir, 'accuracy_comparison.png'),
    )
