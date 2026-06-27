"""
data_loader.py
==============
Bertugas MEMBACA dan MEMBERSIHKAN dataset hasil pertandingan.

Dataset: "International football results from 1872 to present" (Mart Jurisoo).
Kolom: date, home_team, away_team, home_score, away_score, tournament,
       city, country, neutral.

Fungsi utama:
    - load_raw_data()  : baca CSV + validasi kolom (pesan jelas jika error).
    - clean_data()     : parsing tanggal, buang match tanpa skor (match masa
                         depan), urutkan kronologis, dan buat kolom target.
    - load_and_clean() : gabungan keduanya (dipakai modul lain).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    # ketika dijalankan sebagai "python src/xxx.py" (src ada di sys.path)
    import config
except ModuleNotFoundError:  # ketika di-import sebagai package "src.config"
    from src import config

# Kolom yang WAJIB ada pada dataset.
REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]


def load_raw_data(path=None) -> pd.DataFrame:
    """
    Baca file results.csv mentah dan validasi strukturnya.

    Parameters
    ----------
    path : str | Path | None
        Lokasi file CSV. Jika None, pakai default dari config (data/results.csv).

    Returns
    -------
    pd.DataFrame
        DataFrame mentah (belum dibersihkan).

    Raises
    ------
    FileNotFoundError
        Jika file tidak ditemukan -> pesan yang jelas + petunjuk perbaikan.
    ValueError
        Jika ada kolom wajib yang hilang.
    """
    if path is None:
        path = config.RESULTS_CSV

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        # Pesan error yang ramah dan menjelaskan cara memperbaiki.
        raise FileNotFoundError(
            f"\n[ERROR] File dataset tidak ditemukan di: {path}\n"
            f"        Pastikan file 'results.csv' berada di folder 'data/'.\n"
            f"        Dataset bisa diunduh dari Kaggle: 'International football "
            f"results from 1872 to present' (Mart Jurisoo)."
        )

    # Validasi: semua kolom wajib harus ada.
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"\n[ERROR] Kolom berikut tidak ada di dataset: {missing}\n"
            f"        Kolom yang ditemukan: {list(df.columns)}\n"
            f"        Dataset harus punya kolom: {REQUIRED_COLUMNS}"
        )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bersihkan DataFrame mentah dan tambahkan kolom turunan.

    Langkah:
      1. Parsing kolom 'date' menjadi datetime.
      2. Buang baris dengan skor kosong (NA) -> ini match masa depan yang
         belum dimainkan (mis. fixtures World Cup 2026).
      3. Pastikan skor bertipe integer.
      4. Normalisasi kolom 'neutral' menjadi boolean.
      5. Urutkan KRONOLOGIS (penting untuk feature engineering tanpa leakage).
      6. Buat kolom 'result' (target multiclass) dari sudut pandang home team.
    """
    df = df.copy()

    # 1. Tanggal -> datetime. errors='coerce' -> tanggal invalid jadi NaT.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # 2. Skor kosong = match belum dimainkan -> dibuang untuk training.
    #    pd.to_numeric mengubah "NA"/string aneh menjadi NaN.
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])

    # 3. Skor jadi integer.
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # 4. neutral -> boolean (dataset memakai TRUE/FALSE sebagai string).
    df["neutral"] = _to_bool(df["neutral"])

    # 5. Urutkan kronologis. 'kind=stable' menjaga urutan asli untuk tanggal
    #    yang sama -> penting agar feature engineering deterministik.
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    # 6. Target multiclass dari sudut pandang HOME team.
    df["result"] = np.select(
        condlist=[
            df["home_score"] > df["away_score"],   # home menang
            df["home_score"] == df["away_score"],  # seri
        ],
        choicelist=[config.LABEL_HOME_WIN, config.LABEL_DRAW],
        default=config.LABEL_AWAY_WIN,             # away menang
    ).astype(int)

    return df


def load_and_clean(path=None) -> pd.DataFrame:
    """Shortcut: baca + bersihkan dalam satu pemanggilan."""
    return clean_data(load_raw_data(path))


def _to_bool(series: pd.Series) -> pd.Series:
    """Konversi kolom 'neutral' (TRUE/FALSE/1/0/bool) menjadi boolean murni."""
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


if __name__ == "__main__":
    # Smoke test: jalankan "python src/data_loader.py" untuk cek dataset.
    try:
        data = load_and_clean()
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        sys.exit(1)

    print(f"Jumlah match valid : {len(data):,}")
    print(f"Rentang tanggal    : {data['date'].min().date()} s/d "
          f"{data['date'].max().date()}")
    print("\nDistribusi hasil (target):")
    counts = data["result"].value_counts().sort_index()
    for label, n in counts.items():
        print(f"  {label} ({config.LABEL_NAMES[label]:<9}): {n:>7,} "
              f"({n / len(data) * 100:5.1f}%)")
    print("\nContoh 5 baris terakhir:")
    print(data[["date", "home_team", "away_team", "home_score",
                "away_score", "result"]].tail().to_string(index=False))
