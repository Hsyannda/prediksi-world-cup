"""
evaluate.py
===========
Evaluasi model yang sudah dilatih pada TEST set (match 2018 ke atas).

Output:
    - Accuracy & log loss
    - Classification report per kelas (precision/recall/f1)
    - Confusion matrix -> heatmap PNG (reports/confusion_matrix.png)
    - Feature importance top 15 -> PNG (reports/feature_importance.png)
    - Perbandingan dengan baseline naif "selalu prediksi home win"

Jalankan SETELAH train.py:
    python src/evaluate.py
"""

from __future__ import annotations

import sys

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")  # backend non-interaktif (aman tanpa display / di server)
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)

try:
    import config
    from data_loader import load_and_clean
    from feature_engineering import build_feature_table
    from train import temporal_split
except ModuleNotFoundError:
    from src import config
    from src.data_loader import load_and_clean
    from src.feature_engineering import build_feature_table
    from src.train import temporal_split


def _load_model_and_metadata():
    """Muat model + metadata; beri pesan jelas jika belum ada."""
    if not config.MODEL_PATH.exists():
        print(f"[ERROR] Model belum ada di {config.MODEL_PATH}.\n"
              f"        Jalankan dulu: python src/train.py")
        sys.exit(1)
    model = joblib.load(config.MODEL_PATH)
    metadata = joblib.load(config.METADATA_PATH)
    return model, metadata


def plot_confusion_matrix(y_true, y_pred):
    """Buat heatmap confusion matrix dan simpan sebagai PNG."""
    labels = [config.LABEL_HOME_WIN, config.LABEL_DRAW, config.LABEL_AWAY_WIN]
    names = [config.LABEL_NAMES[l] for l in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, cbar=True)
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix (Test Set >= 2018)")
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PNG, dpi=120)
    plt.close()
    print(f"  -> Confusion matrix disimpan: {config.CONFUSION_MATRIX_PNG}")


def plot_feature_importance(model, feature_columns, top_n=15):
    """Plot top-N feature importance dan simpan sebagai PNG."""
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    names = [feature_columns[i] for i in order]
    values = importances[order]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=values, y=names, hue=names, palette="viridis", legend=False)
    plt.xlabel("Importance (gain-based)")
    plt.ylabel("Fitur")
    plt.title(f"Top {top_n} Feature Importance")
    plt.tight_layout()
    plt.savefig(config.FEATURE_IMPORTANCE_PNG, dpi=120)
    plt.close()
    print(f"  -> Feature importance disimpan: {config.FEATURE_IMPORTANCE_PNG}")


def main():
    print("=" * 70)
    print("EVALUASI MODEL")
    print("=" * 70)

    model, metadata = _load_model_and_metadata()
    feature_columns = metadata["feature_columns"]

    # Bangun ulang fitur + split test set yang sama seperti saat training.
    print("\nMembangun fitur & menyiapkan test set ...")
    data = load_and_clean()
    X, y, dates, _ = build_feature_table(data)
    _, X_test, _, y_test = temporal_split(X, y, dates)
    print(f"Test set: {len(X_test):,} match (>= {metadata['split_year']})")

    # Prediksi.
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba, labels=[0, 1, 2])

    print("\n" + "-" * 70)
    print("HASIL MODEL XGBOOST")
    print("-" * 70)
    print(f"Accuracy : {acc:.4f}")
    print(f"Log loss : {ll:.4f}")
    print("\nClassification report per kelas:")
    target_names = [config.LABEL_NAMES[i] for i in (0, 1, 2)]
    print(classification_report(y_test, y_pred, labels=[0, 1, 2],
                                target_names=target_names, digits=3,
                                zero_division=0))

    # --- Baseline naif: selalu prediksi HOME WIN -------------------------
    print("-" * 70)
    print("BASELINE NAIF: selalu prediksi 'Home Win'")
    print("-" * 70)
    y_baseline = np.full_like(y_test, config.LABEL_HOME_WIN)
    baseline_acc = accuracy_score(y_test, y_baseline)
    # Untuk log loss baseline, pakai distribusi kelas pada TRAIN sebagai
    # probabilitas konstan (baseline probabilistik yang wajar).
    X_train, _, y_train, _ = temporal_split(X, y, dates)
    class_freq = (y_train.value_counts(normalize=True)
                  .reindex([0, 1, 2]).fillna(0).values)
    baseline_proba = np.tile(class_freq, (len(y_test), 1))
    baseline_ll = log_loss(y_test, baseline_proba, labels=[0, 1, 2])

    print(f"Accuracy (selalu home win)      : {baseline_acc:.4f}")
    print(f"Log loss (prior distribusi kelas): {baseline_ll:.4f}")

    print("\n" + "-" * 70)
    print("RINGKASAN PERBANDINGAN")
    print("-" * 70)
    print(f"{'Metrik':<12}{'XGBoost':>12}{'Baseline':>12}{'Selisih':>12}")
    print(f"{'Accuracy':<12}{acc:>12.4f}{baseline_acc:>12.4f}"
          f"{acc - baseline_acc:>+12.4f}")
    print(f"{'Log loss':<12}{ll:>12.4f}{baseline_ll:>12.4f}"
          f"{ll - baseline_ll:>+12.4f}  (lebih kecil lebih baik)")

    # --- Plot ------------------------------------------------------------
    print("\nMembuat plot ...")
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, feature_columns, top_n=15)

    print("\n" + "=" * 70)
    print("EVALUASI SELESAI.")
    print("=" * 70)


if __name__ == "__main__":
    main()
