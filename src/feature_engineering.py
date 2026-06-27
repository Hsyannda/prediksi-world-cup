"""
feature_engineering.py
======================
Menghitung fitur untuk setiap match SECARA KRONOLOGIS supaya TIDAK terjadi
data leakage (kebocoran informasi dari masa depan).

Ide intinya:
    Kita mengurutkan seluruh match berdasarkan tanggal, lalu memproses satu per
    satu dari yang TERLAMA ke TERBARU. Untuk setiap match:
        1. Fitur dihitung HANYA dari "state" yang berisi data match-match
           SEBELUMNYA (ELO, form, head-to-head).
        2. SETELAH fitur dicatat, baru state di-update memakai hasil match ini.
    Dengan urutan "hitung fitur -> baru update", model tidak pernah melihat
    informasi dari match itu sendiri maupun match di masa depan.

State yang dipelihara:
    - elo            : dict {team -> rating ELO terkini}
    - recent         : dict {team -> deque berisi N hasil match terakhir}
    - h2h            : dict {frozenset({A,B}) -> list pemenang tiap pertemuan}

Fitur yang dihasilkan (per match):
    1.  elo_home, elo_away          -> ELO rating tiap tim
    2.  elo_diff                    -> selisih ELO (home - away)
    3.  home_winrate, away_winrate  -> win rate 10 match terakhir
    4.  home_avg_scored/conceded    -> rata-rata gol cetak & kebobolan (10 match)
        away_avg_scored/conceded
    5.  h2h_home_winrate            -> head-to-head win rate (sudut pandang home)
    6.  neutral                     -> venue netral atau bukan (0/1)
    7.  cat_worldcup/qualification/ -> tipe turnamen (one-hot)
        friendly/other
    8.  home_avg_goaldiff           -> rata-rata selisih gol (recent form)
        away_avg_goaldiff
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

try:
    import config
except ModuleNotFoundError:
    from src import config

# ---------------------------------------------------------------------------
# DAFTAR KOLOM FITUR (urutan ini HARUS konsisten antara training & prediksi).
# Disimpan bersama model agar kolom prediksi selalu sama dengan saat training.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "elo_home",
    "elo_away",
    "elo_diff",
    "home_winrate",
    "away_winrate",
    "home_avg_scored",
    "home_avg_conceded",
    "away_avg_scored",
    "away_avg_conceded",
    "home_avg_goaldiff",
    "away_avg_goaldiff",
    "h2h_home_winrate",
    "neutral",
    "cat_worldcup",
    "cat_qualification",
    "cat_friendly",
    "cat_other",
]

# Kategori turnamen yang dipakai untuk one-hot encoding.
TOURNAMENT_CATEGORIES = ["worldcup", "qualification", "friendly", "other"]


# ===========================================================================
# 1. UTILITAS TURNAMEN & ELO
# ===========================================================================
def tournament_category(tournament: str) -> str:
    """
    Petakan nama turnamen mentah menjadi salah satu kategori:
        'worldcup' | 'qualification' | 'friendly' | 'other'

    Contoh:
        'FIFA World Cup'               -> worldcup
        'FIFA World Cup qualification' -> qualification
        'UEFA Euro qualification'      -> qualification
        'Friendly'                     -> friendly
        'UEFA Euro', 'Copa America'    -> other
    """
    t = str(tournament).lower()
    if "qualif" in t:
        return "qualification"
    if "world cup" in t:
        return "worldcup"
    if "friendly" in t:
        return "friendly"
    return "other"


def _elo_expected(elo_home: float, elo_away: float, home_adv: float) -> float:
    """
    Probabilitas harapan (expected score) home team menurut formula ELO.
    home_adv menambahkan rating semu untuk keuntungan tuan rumah.
    Hasil di rentang 0..1 (1 = pasti menang, 0.5 = imbang).
    """
    return 1.0 / (1.0 + 10 ** ((elo_away - (elo_home + home_adv)) / 400.0))


def _goal_diff_multiplier(goal_diff: int) -> float:
    """
    Pengali ELO berdasar margin kemenangan (mengikuti World Football Elo).
    Menang telak menggeser rating lebih besar daripada menang tipis.
    """
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0


def _elo_update(elo_home, elo_away, home_score, away_score, neutral, category):
    """
    Hitung ELO baru kedua tim setelah satu match (zero-sum).
    Mengembalikan (elo_home_baru, elo_away_baru).
    """
    home_adv = 0.0 if neutral else config.ELO_HOME_ADVANTAGE
    expected_home = _elo_expected(elo_home, elo_away, home_adv)

    # Skor aktual home: 1 menang, 0.5 seri, 0 kalah.
    if home_score > away_score:
        actual_home = 1.0
    elif home_score == away_score:
        actual_home = 0.5
    else:
        actual_home = 0.0

    k = config.ELO_K_BY_CATEGORY.get(category, config.ELO_K_BY_CATEGORY["other"])
    k_eff = k * _goal_diff_multiplier(home_score - away_score)

    delta = k_eff * (actual_home - expected_home)
    return elo_home + delta, elo_away - delta


# ===========================================================================
# 2. STATE: dipakai untuk membangun fitur secara kronologis
# ===========================================================================
class FeatureState:
    """
    Menyimpan kondisi terkini semua tim saat proses kronologis berjalan.
    Objek ini bisa dibangun dari seluruh histori (lihat build_state) lalu
    dipakai untuk menghitung fitur "saat ini" pada modul prediksi.
    """

    def __init__(self):
        self.elo: dict[str, float] = {}
        # deque per tim berisi dict {scored, conceded, win}
        self.recent: dict[str, deque] = {}
        # head-to-head: frozenset({A,B}) -> list pemenang ('A'/'B'/'draw')
        self.h2h: dict[frozenset, list] = {}

    # --- getter dengan default (cold start) ---------------------------------
    def get_elo(self, team: str) -> float:
        return self.elo.get(team, config.ELO_INITIAL)

    def _recent_for(self, team: str) -> deque:
        if team not in self.recent:
            self.recent[team] = deque(maxlen=config.FORM_WINDOW)
        return self.recent[team]

    def team_form(self, team: str) -> dict:
        """
        Ringkasan form 10 match terakhir sebuah tim:
            winrate, avg_scored, avg_conceded, avg_goaldiff.
        Jika belum ada histori -> pakai nilai default dari config.
        """
        dq = self.recent.get(team)
        if not dq:
            return {
                "winrate": config.DEFAULT_WINRATE,
                "avg_scored": config.DEFAULT_AVG_SCORED,
                "avg_conceded": config.DEFAULT_AVG_CONCEDED,
                "avg_goaldiff": config.DEFAULT_GOAL_DIFF,
            }
        scored = np.mean([m["scored"] for m in dq])
        conceded = np.mean([m["conceded"] for m in dq])
        return {
            "winrate": float(np.mean([m["win"] for m in dq])),
            "avg_scored": float(scored),
            "avg_conceded": float(conceded),
            "avg_goaldiff": float(scored - conceded),
        }

    def h2h_home_winrate(self, home: str, away: str) -> float:
        """
        Win rate head-to-head dari sudut pandang `home` pada pertemuan
        sebelumnya melawan `away`. Seri tidak dihitung sebagai menang.
        """
        meetings = self.h2h.get(frozenset((home, away)))
        if not meetings:
            return config.DEFAULT_H2H_WINRATE
        home_wins = sum(1 for w in meetings if w == home)
        return home_wins / len(meetings)

    # --- update state setelah sebuah match ----------------------------------
    def update(self, home, away, home_score, away_score, neutral, category):
        """Perbarui ELO, form, dan head-to-head memakai hasil 1 match."""
        # ELO
        eh, ea = self.get_elo(home), self.get_elo(away)
        self.elo[home], self.elo[away] = _elo_update(
            eh, ea, home_score, away_score, neutral, category
        )

        # Form (win = 1 jika menang, selain itu 0)
        self._recent_for(home).append(
            {"scored": home_score, "conceded": away_score,
             "win": 1 if home_score > away_score else 0}
        )
        self._recent_for(away).append(
            {"scored": away_score, "conceded": home_score,
             "win": 1 if away_score > home_score else 0}
        )

        # Head-to-head
        if home_score > away_score:
            winner = home
        elif away_score > home_score:
            winner = away
        else:
            winner = "draw"
        self.h2h.setdefault(frozenset((home, away)), []).append(winner)


# ===========================================================================
# 3. HITUNG FITUR SATU MATCH (fungsi murni, baca state saja)
# ===========================================================================
def compute_match_features(home, away, neutral, category, state):
    """
    Bentuk satu baris fitur (dict) untuk satu match memakai kondisi `state`
    saat ini. Fungsi ini TIDAK mengubah state -> dipakai sama persis di
    training maupun prediksi.
    """
    elo_home = state.get_elo(home)
    elo_away = state.get_elo(away)
    form_home = state.team_form(home)
    form_away = state.team_form(away)

    # one-hot kategori turnamen
    onehot = {f"cat_{c}": 0 for c in TOURNAMENT_CATEGORIES}
    onehot[f"cat_{category}"] = 1

    feats = {
        "elo_home": elo_home,
        "elo_away": elo_away,
        "elo_diff": elo_home - elo_away,
        "home_winrate": form_home["winrate"],
        "away_winrate": form_away["winrate"],
        "home_avg_scored": form_home["avg_scored"],
        "home_avg_conceded": form_home["avg_conceded"],
        "away_avg_scored": form_away["avg_scored"],
        "away_avg_conceded": form_away["avg_conceded"],
        "home_avg_goaldiff": form_home["avg_goaldiff"],
        "away_avg_goaldiff": form_away["avg_goaldiff"],
        "h2h_home_winrate": state.h2h_home_winrate(home, away),
        "neutral": int(bool(neutral)),
    }
    feats.update(onehot)
    return feats


# ===========================================================================
# 4. BANGUN TABEL FITUR (untuk training) & STATE (untuk prediksi)
# ===========================================================================
def build_feature_table(df: pd.DataFrame):
    """
    Proses seluruh match KRONOLOGIS dan hasilkan tabel fitur lengkap.

    Asumsi: df sudah dibersihkan & terurut tanggal (lihat data_loader).

    Returns
    -------
    X : pd.DataFrame  -> fitur (kolom sesuai FEATURE_COLUMNS)
    y : pd.Series     -> target (kolom 'result')
    dates : pd.Series -> tanggal tiap match (untuk split temporal)
    state : FeatureState -> kondisi akhir setelah semua match diproses
    """
    state = FeatureState()
    rows = []

    # itertuples jauh lebih cepat daripada iterrows untuk ~50k baris.
    for r in df.itertuples(index=False):
        category = tournament_category(r.tournament)

        # 1) Hitung fitur DULU (hanya dari data sebelum match ini).
        feats = compute_match_features(
            r.home_team, r.away_team, r.neutral, category, state
        )
        rows.append(feats)

        # 2) BARU update state dengan hasil match ini (cegah leakage).
        state.update(
            r.home_team, r.away_team, r.home_score, r.away_score,
            r.neutral, category,
        )

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    y = df["result"].reset_index(drop=True)
    dates = df["date"].reset_index(drop=True)
    return X, y, dates, state


def build_state(df: pd.DataFrame) -> FeatureState:
    """
    Bangun HANYA state akhir dari seluruh histori (tanpa membentuk tabel
    fitur). Dipakai modul prediksi untuk mendapatkan kondisi terkini tiap tim.
    """
    state = FeatureState()
    for r in df.itertuples(index=False):
        category = tournament_category(r.tournament)
        state.update(
            r.home_team, r.away_team, r.home_score, r.away_score,
            r.neutral, category,
        )
    return state


if __name__ == "__main__":
    # Smoke test: bangun fitur dan tampilkan beberapa baris.
    try:
        from data_loader import load_and_clean
    except ModuleNotFoundError:
        from src.data_loader import load_and_clean

    data = load_and_clean()
    X, y, dates, _ = build_feature_table(data)
    print(f"Bentuk tabel fitur : {X.shape}")
    print(f"Jumlah fitur       : {len(FEATURE_COLUMNS)}")
    print("\nContoh 5 baris fitur terakhir:")
    print(X.tail().to_string())
