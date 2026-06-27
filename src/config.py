"""
config.py
=========
Konfigurasi terpusat untuk seluruh project: lokasi file, parameter ELO,
batas split temporal, dan daftar kolom fitur.

Semua path dihitung relatif terhadap ROOT project (bukan terhadap current
working directory) supaya script bisa dijalankan dari mana saja, misal:
    python src/train.py
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# PATH / LOKASI FILE
# ---------------------------------------------------------------------------
# parents[1] = folder root project (folder di atas src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RESULTS_CSV = DATA_DIR / "results.csv"

# Artefak hasil training
MODEL_PATH = MODELS_DIR / "xgb_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.joblib"  # daftar fitur + info lain

# Output evaluasi
CONFUSION_MATRIX_PNG = REPORTS_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PNG = REPORTS_DIR / "feature_importance.png"

# Pastikan folder output selalu ada
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# TARGET / LABEL
# ---------------------------------------------------------------------------
# Target multiclass dari sudut pandang HOME team (tim pertama).
# Untuk pertandingan netral (World Cup), "home" hanya berarti tim pertama.
LABEL_HOME_WIN = 0   # home menang
LABEL_DRAW = 1       # seri
LABEL_AWAY_WIN = 2   # away menang

LABEL_NAMES = {
    LABEL_HOME_WIN: "Home Win",
    LABEL_DRAW: "Draw",
    LABEL_AWAY_WIN: "Away Win",
}

# ---------------------------------------------------------------------------
# SPLIT TEMPORAL
# ---------------------------------------------------------------------------
# Data sebelum tahun ini -> TRAIN, tahun ini ke atas -> TEST.
TEST_SPLIT_YEAR = 2018

# ---------------------------------------------------------------------------
# PARAMETER FEATURE ENGINEERING
# ---------------------------------------------------------------------------
ELO_INITIAL = 1500.0       # rating awal setiap tim
ELO_HOME_ADVANTAGE = 65.0  # bonus rating semu untuk tuan rumah (0 jika netral)
FORM_WINDOW = 10           # jumlah match terakhir untuk menghitung "form"

# K-factor ELO berdasar kepentingan turnamen (semakin penting, semakin besar).
ELO_K_BY_CATEGORY = {
    "worldcup": 50.0,
    "qualification": 35.0,
    "friendly": 20.0,
    "other": 30.0,
}

# Nilai default ketika sebuah tim belum punya histori (cold start).
DEFAULT_WINRATE = 0.5
DEFAULT_AVG_SCORED = 1.3     # rata-rata gol per match secara umum
DEFAULT_AVG_CONCEDED = 1.3
DEFAULT_GOAL_DIFF = 0.0
DEFAULT_H2H_WINRATE = 0.5
