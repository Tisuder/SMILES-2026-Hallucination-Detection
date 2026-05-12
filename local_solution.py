import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from probe import HallucinationProbe

def run_local_evaluation():
    """
    Performs 5-fold cross-validation for different feature extraction configurations.
    Prints individual fold results and final mean metrics for stability analysis.
    """
    # Feature prefixes representing different research iterations
    prefixes = ['', '_40', '_hot', '_geom', '_20']
    data_dir = Path("data")

    for pref in prefixes:
        label = pref if pref != '' else "base"
        print(f"\n{'='*15} EXPERIMENT: {label} {'='*15}")

        try:
            # Load pre-extracted features and invariant labels
            X = np.load(data_dir / f"features_train{pref}.npy")
            y = np.load(data_dir / "labels_train.npy")
        except FileNotFoundError:
            print(f"Skipping {label}: Required .npy files not found in {data_dir}")
            continue

        print(f"Input dimension: {X.shape}")
        print("Starting 5-Fold Cross-Validation...\n")

        # Initialize stratified k-fold to maintain class distribution
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metrics = {"auroc": [], "acc": [], "f1": []}

        start_time = time.time()
        for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Initialize and fit the student probe
            probe = HallucinationProbe()
            probe.fit(X_train, y_train)
            
            # Tune decision threshold on the current validation fold
            probe.fit_hyperparameters(X_val, y_val)

            # Evaluate on validation fold
            probs = probe.predict_proba(X_val)[:, 1]
            preds = probe.predict(X_val)
            
            # Calculate metrics
            val_auroc = roc_auc_score(y_val, probs)
            val_acc = accuracy_score(y_val, preds)
            val_f1 = f1_score(y_val, preds, zero_division=0)

            metrics["auroc"].append(val_auroc)
            metrics["acc"].append(val_acc)
            metrics["f1"].append(val_f1)
            
            print(f"Fold {i+1}: Val AUROC = {val_auroc:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        # Summary statistics
        print("-" * 50)
        print(f"MEAN VAL AUROC : {np.mean(metrics['auroc']):.4f}")
        print(f"MEAN ACCURACY  : {np.mean(metrics['acc']):.4f}")
        print(f"MEAN F1 SCORE  : {np.mean(metrics['f1']):.4f}")
        print(f"Total duration : {time.time() - start_time:.2f}s")
        print("-" * 50)

if __name__ == '__main__':
    run_local_evaluation()