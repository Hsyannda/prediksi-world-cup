# ⚽ Prediksi Hasil Pertandingan Sepak Bola Internasional (XGBoost)

Project machine learning untuk memprediksi hasil pertandingan sepak bola
internasional / World Cup menggunakan **XGBoost**. Model mengeluarkan
probabilitas tiga kemungkinan hasil dari sudut pandang tim pertama (*home*):

| Label | Arti        |
|:-----:|-------------|
| `0`   | Home menang |
| `1`   | Seri        |
| `2`   | Away menang |

> Untuk pertandingan netral (World Cup), **"home" hanya berarti tim pertama**.

Dataset yang dipakai: **"International football results from 1872 to present"**
oleh *Mart Jürisoo* (`data/results.csv`, ~48 ribu pertandingan).

---

## 📂 Struktur Project

```
Prediksi World Cup/
├── data/
│   └── results.csv              # dataset asli (date, home_team, away_team, ...)
├── models/                      # artefak hasil training (model + metadata)
│   ├── xgb_model.joblib
│   └── model_metadata.joblib
├── reports/                     # output evaluasi (PNG)
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── src/
│   ├── config.py                # konfigurasi terpusat (path, ELO, split, dll)
│   ├── data_loader.py           # baca + validasi + bersihkan dataset
│   ├── feature_engineering.py   # fitur kronologis tanpa data leakage
│   ├── train.py                 # training + hyperparameter tuning
│   ├── evaluate.py              # evaluasi + plot + baseline
│   └── predict.py               # prediksi match + simulasi bracket
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

Python 3.11 · pandas · numpy · scikit-learn · xgboost · matplotlib · seaborn · joblib

---

## 🚀 Instalasi

```bash
# 1. (opsional) buat virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate

# 2. install dependency
pip install -r requirements.txt
```

Pastikan file dataset berada di `data/results.csv`. Jika tidak ada, semua
script akan menampilkan pesan error yang jelas beserta petunjuk perbaikan.

Cek dataset (opsional):

```bash
python src/data_loader.py
```

---

## 🏋️ Training

```bash
python src/train.py
```

Yang dilakukan:

1. Memuat & membersihkan data (membuang fixtures yang belum dimainkan).
2. Membangun fitur **secara kronologis** untuk menghindari *data leakage*.
3. **Split temporal**: `train = sebelum 2018`, `test = 2018 ke atas`.
4. **Hyperparameter tuning** via `RandomizedSearchCV` + `TimeSeriesSplit`
   (`max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`).
5. Menyimpan model terbaik + metadata ke `models/`.

### Kenapa split temporal, bukan random?

Data ini **time-series**: kekuatan tim berubah sepanjang waktu dan fitur
(ELO, *form*) dihitung dari masa lalu. Dengan random split, sebagian match
masa depan bisa bocor ke data train sehingga skor evaluasi jadi terlalu
optimis. Split temporal meniru kondisi nyata: **latih dengan masa lalu, uji
pada masa depan**.

---

## 📊 Evaluasi

```bash
python src/evaluate.py
```

Menghasilkan:

- **Accuracy** & **log loss** pada test set.
- **Classification report** per kelas (precision / recall / f1).
- **Confusion matrix** → `reports/confusion_matrix.png`.
- **Feature importance** top 15 → `reports/feature_importance.png`.
- Perbandingan dengan **baseline naif** ("selalu prediksi home win").

---

## 🔮 Prediksi

Fungsi inti: `predict_match(home_team, away_team, neutral=True)` menghitung
fitur terkini kedua tim dari data historis, lalu mengembalikan
`P(home menang)`, `P(seri)`, `P(away menang)`.

### Via CLI

```bash
# Pertandingan netral (World Cup)
python src/predict.py --home "Argentina" --away "France" --neutral

# Pertandingan dengan tuan rumah (non-netral)
python src/predict.py --home "Brazil" --away "Germany"
```

Contoh output:

```
========================================================
  Argentina  vs  France
  Venue: Netral (World Cup)
  ELO  : Argentina 2120  |  France 2050
========================================================
  Argentina menang        41.30%  |############------------------|
  Seri                    26.10%  |########----------------------|
  France menang           32.60%  |##########--------------------|
--------------------------------------------------------
  Prediksi : HOME WIN
========================================================
```

### Via Python

```python
from src.predict import predict_match

res = predict_match("Argentina", "France", neutral=True)
print(res["p_home_win"], res["p_draw"], res["p_away_win"])
```

---

## 🏆 Bonus: Simulasi Bracket Knockout

Mensimulasikan fase gugur (single elimination) berdasarkan probabilitas model.
Pada laga gugur, seri dianggap berakhir adu penalti
(`P(lolos) = P(menang) + 0.5 × P(seri)`). Peluang juara dihitung via
**Monte Carlo**.

```bash
python src/predict.py --bracket "Argentina,France,Brazil,England" --sims 2000
```

Jumlah tim harus pangkat 2 (`4`, `8`, `16`). Pasangan awal mengikuti urutan:
`(tim1 vs tim2)`, `(tim3 vs tim4)`, dst.

```python
from src.predict import simulate_knockout

peluang_juara = simulate_knockout(
    ["Argentina", "France", "Brazil", "England"], n_sims=5000
)
```

---

## 🧠 Feature Engineering (tanpa data leakage)

Untuk setiap match, fitur dihitung **hanya dari data sebelum tanggal match
tersebut**. Seluruh match diproses kronologis (terlama → terbaru): fitur
dicatat **lebih dulu**, baru *state* di-update dengan hasil match itu.

| # | Fitur | Keterangan |
|---|-------|-----------|
| 1 | `elo_home`, `elo_away` | ELO rating tiap tim (mulai 1500, update tiap match) |
| 2 | `elo_diff` | Selisih ELO (home − away) |
| 3 | `home_winrate`, `away_winrate` | Win rate 10 match terakhir |
| 4 | `*_avg_scored`, `*_avg_conceded` | Rata-rata gol cetak & kebobolan (10 match) |
| 5 | `h2h_home_winrate` | Head-to-head win rate (sudut pandang home) |
| 6 | `neutral` | Venue netral (1) atau bukan (0) |
| 7 | `cat_worldcup/qualification/friendly/other` | Tipe turnamen (one-hot) |
| 8 | `home_avg_goaldiff`, `away_avg_goaldiff` | Rata-rata selisih gol (recent form) |

**Detail ELO:** memakai keuntungan tuan rumah (`+65` rating semu jika tidak
netral), pengali margin gol (menang telak menggeser rating lebih besar), dan
K-factor berbeda per kepentingan turnamen (World Cup > kualifikasi > friendly).

---

## ✅ Urutan Menjalankan

```bash
pip install -r requirements.txt   # 1. install
python src/train.py               # 2. latih model (buat models/)
python src/evaluate.py            # 3. evaluasi (buat reports/*.png)
python src/predict.py --home "Argentina" --away "France" --neutral   # 4. prediksi
```

---

## 📝 Catatan

- Komentar kode ditulis dalam **Bahasa Indonesia** agar mudah dipelajari.
- Semua path dihitung relatif terhadap root project, jadi script bisa
  dijalankan dari folder mana pun.
- Model bersifat **probabilistik** — sepak bola penuh kejutan; gunakan
  probabilitas sebagai estimasi, bukan kepastian. 🙂
