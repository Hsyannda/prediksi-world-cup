"""
train.py
========
Melatih model XGBoost untuk memprediksi hasil pertandingan (multiclass).

Alur:
    1. Load + bersihkan data (data_loader).
    2. Bangun fitur kronologis tanpa leakage (feature_engineering).
    3. SPLIT TEMPORAL (bukan random): train = sebelum 2018, test = 2018 ke atas.
    4. Hyperparameter tuning dengan RandomizedSearchCV + TimeSeriesSplit.
    5. Simpan model terbaik + metadata (daftar fitur) ke folder models/.

------------------------------------------------------------------------------
KENAPA SPLIT TEMPORAL, BUKAN RANDOM?
------------------------------------------------------------------------------
Data ini bersifat time-series: kekuatan tim berubah sepanjang waktu, dan fitur
seperti ELO/form dihitung dari masa lalu. Kalau kita memakai random split,
sebagian match masa DEPAN bisa masuk ke data train sementara match masa LALU
masuk ke test. Akibatnya model "mengintip" masa depan -> skor evaluasi jadi
terlalu optimis dan tidak mencerminkan kondisi nyata (memprediksi pertandingan
yang belum terjadi). Split temporal meniru situasi produksi sebenarnya:
latih dengan masa lalu, uji pada masa depan.
"""

from __future__ import annotations

import time

import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

try:
    import config
    from data_loader import load_and_clean
    from feature_engineering import FEATURE_COLUMNS, build_feature_table
except ModuleNotFoundError:
    from src import config
    from src.data_loader import load_and_clean
    from src.feature_engineering import FEATURE_COLUMNS, build_feature_table


def temporal_split(X, y, dates, split_year=config.TEST_SPLIT_YEAR):
    """
    Pisahkan data berdasar tahun:
        train = match dengan tahun < split_year
        test  = match dengan tahun >= split_year
    """
    is_train = dates.dt.year < split_year
    X_train, y_train = X[is_train], y[is_train]
    X_test, y_test = X[~is_train], y[~is_train]
    return X_train, X_test, y_train, y_test


def build_search(random_state=42):
    """
    Siapkan RandomizedSearchCV untuk men-tuning hyperparameter XGBoost.

    Catatan:
      - objective='multi:softprob' -> output probabilitas tiap kelas.
      - TimeSeriesSplit dipakai sebagai CV agar fold validasi selalu berada
        SETELAH fold training (konsisten dengan sifat time-series, hindari
        leakage juga di dalam proses tuning).
    """
    base_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",      # cepat untuk dataset menengah
        random_state=random_state,
        n_jobs=-1,
    )

    # Ruang pencarian hyperparameter (sesuai permintaan).
    param_distributions = {
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
        "n_estimators": [200, 300, 400, 500, 600, 800],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    }

    tscv = TimeSeriesSplit(n_splits=4)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=25,                 # jumlah kombinasi yang dicoba
        scoring="neg_log_loss",    # fokus ke kualitas probabilitas
        cv=tscv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    return search


def main():
    print("=" * 70)
    print("TRAINING MODEL XGBOOST - PREDIKSI HASIL SEPAK BOLA")
    print("=" * 70)

    # 1-2. Load data + bangun fitur kronologis.
    print("\n[1/4] Memuat & membersihkan data ...")
    data = load_and_clean()
    print(f"      {len(data):,} match valid "
          f"({data['date'].min().date()} s/d {data['date'].max().date()})")

    print("[2/4] Membangun fitur secara kronologis (tanpa data leakage) ...")
    t0 = time.time()
    X, y, dates, _ = build_feature_table(data)
    print(f"      Selesai dalam {time.time() - t0:.1f}s -> fitur: {X.shape}")

    # 3. Split temporal.
    X_train, X_test, y_train, y_test = temporal_split(X, y, dates)
    print(f"[3/4] Split temporal di tahun {config.TEST_SPLIT_YEAR}:")
    print(f"      Train (< {config.TEST_SPLIT_YEAR}) : {len(X_train):,} match")
    print(f"      Test  (>= {config.TEST_SPLIT_YEAR}): {len(X_test):,} match")

    # 4. Tuning + training.
    print("[4/4] Hyperparameter tuning (RandomizedSearchCV) ...")
    search = build_search()
    t0 = time.time()
    search.fit(X_train, y_train)
    print(f"      Tuning selesai dalam {time.time() - t0:.1f}s")
    print(f"      Best CV log loss : {-search.best_score_:.4f}")
    print(f"      Best params      : {search.best_params_}")

    best_model = search.best_estimator_

    # Skor cepat di test set (evaluasi lengkap ada di evaluate.py).
    test_acc = best_model.score(X_test, y_test)
    print(f"      Akurasi test set : {test_acc:.4f}")

    # 5. Simpan model + metadata.
    joblib.dump(best_model, config.MODEL_PATH)
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "best_params": search.best_params_,
        "best_cv_logloss": float(-search.best_score_),
        "test_accuracy": float(test_acc),
        "split_year": config.TEST_SPLIT_YEAR,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "trained_on": str(np.datetime64("now")),
    }
    joblib.dump(metadata, config.METADATA_PATH)

    print("\n" + "=" * 70)
    print("SELESAI. Artefak tersimpan:")
    print(f"  - Model    : {config.MODEL_PATH}")
    print(f"  - Metadata : {config.METADATA_PATH}")
    print("Jalankan 'python src/evaluate.py' untuk evaluasi lengkap.")
    print("=" * 70)


if __name__ == "__main__":
    main()
