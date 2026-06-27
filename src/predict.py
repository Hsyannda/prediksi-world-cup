"""
predict.py
==========
Memprediksi hasil satu pertandingan memakai model XGBoost yang sudah dilatih.

Fungsi inti:
    predict_match(home_team, away_team, neutral=True)
        -> mengembalikan probabilitas (P_home_win, P_draw, P_away_win)
           dengan menghitung fitur TERKINI kedua tim dari data historis.

CLI:
    python src/predict.py --home "Argentina" --away "France" --neutral
    python src/predict.py --home "Brazil" --away "Germany"          (non-netral)

BONUS:
    simulate_knockout(...) -> simulasi bracket fase gugur World Cup
    python src/predict.py --bracket "Argentina,France,Brazil,England"
"""

from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd

try:
    import config
    from data_loader import load_and_clean
    from feature_engineering import (
        compute_match_features,
        build_state,
        tournament_category,
    )
except ModuleNotFoundError:
    from src import config
    from src.data_loader import load_and_clean
    from src.feature_engineering import (
        compute_match_features,
        build_state,
        tournament_category,
    )


# Cache modul-level agar data & state hanya dibangun sekali per proses
# (berguna saat simulasi bracket memanggil predict_match berkali-kali).
_CACHE = {"model": None, "metadata": None, "state": None, "teams": None}


def _ensure_loaded():
    """Muat model, metadata, dan state historis (sekali saja)."""
    if _CACHE["model"] is not None:
        return

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model belum ada di {config.MODEL_PATH}. "
            f"Jalankan dulu: python src/train.py"
        )

    _CACHE["model"] = joblib.load(config.MODEL_PATH)
    _CACHE["metadata"] = joblib.load(config.METADATA_PATH)

    data = load_and_clean()
    _CACHE["state"] = build_state(data)
    # Set nama tim valid (untuk validasi & fuzzy matching).
    _CACHE["teams"] = set(data["home_team"]) | set(data["away_team"])


def _resolve_team(name: str) -> str:
    """
    Cocokkan nama tim yang diinput user dengan nama di dataset.
    Mendukung pencocokan persis (case-insensitive) dan substring sederhana.
    """
    teams = _CACHE["teams"]
    if name in teams:
        return name
    # case-insensitive exact
    lower_map = {t.lower(): t for t in teams}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    # substring (mis. "korea" -> ada beberapa, ambil yang paling pendek/cocok)
    candidates = [t for t in teams if name.lower() in t.lower()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"Nama tim '{name}' ambigu, beberapa kandidat: "
            f"{sorted(candidates)[:8]}. Mohon lebih spesifik."
        )
    raise ValueError(
        f"Tim '{name}' tidak ditemukan di dataset. "
        f"Periksa ejaan (mis. 'United States', 'South Korea')."
    )


def predict_match(home_team, away_team, neutral=True, tournament="FIFA World Cup"):
    """
    Prediksi probabilitas hasil satu match dari sudut pandang home_team.

    Parameters
    ----------
    home_team, away_team : str
        Nama tim (akan dicocokkan dengan dataset).
    neutral : bool
        True jika venue netral (default untuk World Cup).
    tournament : str
        Jenis turnamen (default 'FIFA World Cup' -> kategori one-hot worldcup).

    Returns
    -------
    dict dengan kunci:
        home, away, neutral,
        p_home_win, p_draw, p_away_win,
        predicted_label, predicted_name,
        elo_home, elo_away
    """
    _ensure_loaded()
    model = _CACHE["model"]
    state = _CACHE["state"]
    feature_columns = _CACHE["metadata"]["feature_columns"]

    home = _resolve_team(home_team)
    away = _resolve_team(away_team)
    category = tournament_category(tournament)

    # Hitung fitur terkini dari state historis (tanpa mengubah state).
    feats = compute_match_features(home, away, neutral, category, state)
    X = pd.DataFrame([feats], columns=feature_columns)

    proba = model.predict_proba(X)[0]
    label = int(np.argmax(proba))

    return {
        "home": home,
        "away": away,
        "neutral": bool(neutral),
        "p_home_win": float(proba[config.LABEL_HOME_WIN]),
        "p_draw": float(proba[config.LABEL_DRAW]),
        "p_away_win": float(proba[config.LABEL_AWAY_WIN]),
        "predicted_label": label,
        "predicted_name": config.LABEL_NAMES[label],
        "elo_home": float(state.get_elo(home)),
        "elo_away": float(state.get_elo(away)),
    }


def print_prediction(res: dict):
    """Tampilkan hasil prediksi dengan rapi di terminal."""
    venue = "Netral (World Cup)" if res["neutral"] else f"Kandang {res['home']}"
    bar_w = 30

    def bar(p):
        filled = int(round(p * bar_w))
        return "#" * filled + "-" * (bar_w - filled)

    print("\n" + "=" * 56)
    print(f"  {res['home']}  vs  {res['away']}")
    print(f"  Venue: {venue}")
    print(f"  ELO  : {res['home']} {res['elo_home']:.0f}  |  "
          f"{res['away']} {res['elo_away']:.0f}")
    print("=" * 56)
    rows = [
        (f"{res['home']} menang", res["p_home_win"]),
        ("Seri", res["p_draw"]),
        (f"{res['away']} menang", res["p_away_win"]),
    ]
    for label, p in rows:
        print(f"  {label:<22} {p*100:6.2f}%  |{bar(p)}|")
    print("-" * 56)
    print(f"  Prediksi : {res['predicted_name'].upper()}")
    print("=" * 56 + "\n")


# ===========================================================================
# BONUS: SIMULASI BRACKET KNOCKOUT WORLD CUP
# ===========================================================================
def _advance_probability(res: dict) -> float:
    """
    Peluang HOME (tim pertama) lolos di laga sistem gugur.
    Pada fase gugur seri berakhir adu penalti -> kita bagi rata peluang seri.
        P(home lolos) = P(home menang) + 0.5 * P(seri)
    """
    return res["p_home_win"] + 0.5 * res["p_draw"]


def simulate_single_match(team_a, team_b, rng=None, sample=True):
    """
    Tentukan pemenang satu laga gugur antara team_a vs team_b (venue netral).
    - sample=True  : ambil pemenang secara acak sesuai probabilitas (Monte Carlo)
    - sample=False : pilih tim dengan peluang lolos tertinggi (deterministik)
    """
    res = predict_match(team_a, team_b, neutral=True)
    p_a = _advance_probability(res)  # peluang team_a (home) lolos
    if sample:
        rng = rng or np.random.default_rng()
        return team_a if rng.random() < p_a else team_b
    return team_a if p_a >= 0.5 else team_b


def simulate_knockout(teams, n_sims=2000, seed=42, verbose=True):
    """
    Simulasi bracket fase gugur (single elimination).

    Parameters
    ----------
    teams : list[str]
        Daftar tim sesuai urutan bracket. Jumlah harus pangkat 2
        (mis. 4, 8, 16). Pasangan awal: (0 vs 1), (2 vs 3), dst.
    n_sims : int
        Banyaknya simulasi Monte Carlo untuk memperkirakan peluang juara.
    seed : int
        Seed RNG agar hasil reproducible.
    verbose : bool
        Cetak satu contoh jalannya bracket (deterministik) + peluang juara.

    Returns
    -------
    dict {team -> peluang menjadi juara}
    """
    _ensure_loaded()
    teams = [_resolve_team(t) for t in teams]

    n = len(teams)
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(
            f"Jumlah tim harus pangkat 2 (2,4,8,16,...). Diberikan: {n}."
        )

    # 1) Monte Carlo: hitung frekuensi juara.
    rng = np.random.default_rng(seed)
    champion_count = {t: 0 for t in teams}
    for _ in range(n_sims):
        bracket = list(teams)
        while len(bracket) > 1:
            bracket = [
                simulate_single_match(bracket[i], bracket[i + 1], rng, sample=True)
                for i in range(0, len(bracket), 2)
            ]
        champion_count[bracket[0]] += 1

    champion_prob = {t: champion_count[t] / n_sims for t in teams}

    if verbose:
        # 2) Satu contoh bracket "jalur paling mungkin" (deterministik).
        print("\n" + "=" * 56)
        print("  SIMULASI BRACKET KNOCKOUT (jalur paling mungkin)")
        print("=" * 56)
        bracket = list(teams)
        round_no = 1
        while len(bracket) > 1:
            print(f"\n  -- Round {round_no} ({len(bracket)} tim) --")
            winners = []
            for i in range(0, len(bracket), 2):
                a, b = bracket[i], bracket[i + 1]
                res = predict_match(a, b, neutral=True)
                w = a if _advance_probability(res) >= 0.5 else b
                print(f"    {a:<18} vs {b:<18} -> {w}  "
                      f"(P_lolos {a}={_advance_probability(res)*100:.1f}%)")
                winners.append(w)
            bracket = winners
            round_no += 1
        print(f"\n  >> Pemenang jalur deterministik: {bracket[0]}")

        print("\n" + "-" * 56)
        print(f"  PELUANG JUARA (Monte Carlo, {n_sims} simulasi)")
        print("-" * 56)
        for t, p in sorted(champion_prob.items(), key=lambda x: -x[1]):
            bar = "#" * int(round(p * 30))
            print(f"    {t:<20} {p*100:6.2f}%  |{bar}")
        print("=" * 56 + "\n")

    return champion_prob


# ===========================================================================
# CLI
# ===========================================================================
def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Prediksi hasil pertandingan sepak bola (XGBoost).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Contoh:\n"
            '  python src/predict.py --home "Argentina" --away "France" --neutral\n'
            '  python src/predict.py --home "Brazil" --away "Germany"\n'
            '  python src/predict.py --bracket "Argentina,France,Brazil,England"\n'
        ),
    )
    p.add_argument("--home", help="Nama tim home (tim pertama).")
    p.add_argument("--away", help="Nama tim away (tim kedua).")
    p.add_argument("--neutral", action="store_true",
                   help="Venue netral (default untuk World Cup).")
    p.add_argument("--bracket",
                   help="Daftar tim dipisah koma untuk simulasi knockout "
                        "(jumlah harus 2/4/8/16).")
    p.add_argument("--sims", type=int, default=2000,
                   help="Jumlah simulasi Monte Carlo untuk --bracket.")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)

    try:
        if args.bracket:
            teams = [t.strip() for t in args.bracket.split(",") if t.strip()]
            simulate_knockout(teams, n_sims=args.sims)
            return

        if args.home and args.away:
            res = predict_match(args.home, args.away, neutral=args.neutral)
            print_prediction(res)
            return

        # Tidak ada argumen yang cukup -> tampilkan bantuan.
        _build_arg_parser().print_help()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
