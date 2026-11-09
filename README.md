# Prediksi Suhu Udara 1 Jam ke Depan dengan XGBoost

Dokumentasi untuk notebook `deteksi_suhu_cuaca_1_jam_kedepan_menggunakan_xgboost` (Google Colab).

---

## 1. Ringkasan

| Item | Keterangan |
|---|---|
| **Tujuan** | Memprediksi suhu udara (`T (degC)`) **60 menit ke depan** |
| **Jenis masalah** | Regresi *time series* tabular |
| **Model** | `xgboost.XGBRegressor` |
| **Dataset** | Jena Climate 2009–2016 (Max Planck Institute, Jerman), interval pengamatan **10 menit** |
| **Sumber data** | `https://huggingface.co/datasets/sayanroy058/Jena-Climate/resolve/main/jena_climate_2009_2016.csv` |
| **Metrik evaluasi** | MAE, MSE, RMSE, R² |
| **Pembagian data** | Temporal 80% latih / 20% uji (tanpa pengacakan) |

---

## 2. Dependensi

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

| Library | Fungsi |
|---|---|
| `pandas`, `numpy` | Manipulasi data dan perhitungan numerik |
| `matplotlib`, `seaborn` | Visualisasi EDA dan hasil model |
| `scikit-learn` | Metrik evaluasi (`mean_absolute_error`, `mean_squared_error`, `r2_score`) |
| `xgboost` | Model regresi *gradient boosting* |

---

## 3. Alur Kerja (Pipeline)

```
Load data → Audit kualitas → Preprocessing → Feature engineering
   → EDA → Split temporal → Training XGBoost → Learning curve
   → Evaluasi metrik → Tabel sampel prediksi
```

### 3.1 Load Data & Audit Kualitas Awal

Data dibaca langsung dari URL dengan `pd.read_csv`. Audit awal mencakup:

- Dimensi data mentah (baris, kolom)
- Jumlah baris duplikat (`df.duplicated().sum()`)
- Jumlah *missing value* per kolom
- Anomali sensor: nilai `wv (m/s)` negatif (kode error `-9999`)

### 3.2 Preprocessing

| Langkah | Detail |
|---|---|
| Parsing waktu | `Date Time` dikonversi ke `datetime` dengan format `%d.%m.%Y %H:%M:%S` |
| Pengurutan | Data diurutkan kronologis berdasarkan `Date Time` |
| Penanganan anomali | `wv (m/s)` < 0 diganti dengan `0.0` |
| Pembuatan target | `target_suhu_next_hour = T (degC).shift(-6)` |
| Pembersihan NaN | `dropna()` untuk baris yang kosong akibat *lag* dan *shift* target |

**Mengapa `shift(-6)`?** Interval data 10 menit, sehingga 6 langkah = 60 menit ke depan.

### 3.3 Feature Engineering

**a. Cyclical encoding** (waktu bersifat siklik, jam 23 dekat dengan jam 0):

| Fitur | Rumus |
|---|---|
| `hour_sin`, `hour_cos` | sin/cos dari `2π × jam / 24` |
| `month_sin`, `month_cos` | sin/cos dari `2π × bulan / 12` |

**b. Lag features** untuk 5 variabel: `T (degC)`, `rh (%)`, `p (mbar)`, `wv (m/s)`, `VPdef (mbar)`.

| Lag | Langkah | Arti waktu |
|---|---|---|
| `_lag_1` | 1 | 10 menit sebelumnya |
| `_lag_3` | 3 | 30 menit sebelumnya |
| `_lag_6` | 6 | 60 menit sebelumnya |

Total: 5 variabel × 3 lag = **15 fitur lag**.

**c. Fitur akhir** = semua kolom kecuali `Date Time` dan `target_suhu_next_hour` (kolom sensor asli + 4 fitur siklik + 15 fitur lag).

### 3.4 Exploratory Data Analysis (EDA)

| Jenis | Visualisasi | Tujuan |
|---|---|---|
| Univariat | Histplot + KDE target suhu | Melihat distribusi target |
| Univariat | Histplot tekanan udara | Distribusi `p (mbar)` |
| Univariat | Boxplot kelembapan | Sebaran dan outlier `rh (%)` |
| Bivariat | Scatter `T (degC)` vs target | Hubungan suhu saat ini dan 1 jam ke depan |
| Bivariat | Scatter `rh (%)` vs target | Hubungan kelembapan dan target |
| Multivariat | Heatmap korelasi Pearson | Korelasi fitur, lag, dan target |

Visualisasi bivariat dan multivariat memakai sub-sampel acak 5.000 baris (`random_state=42`) agar cepat dirender.

### 3.5 Data Splitting

Pembagian **temporal** (bukan acak) agar tidak terjadi kebocoran data dari masa depan ke data latih:

```python
split_idx = int(len(df_clean) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
```

### 3.6 Model & Hyperparameter

```python
xgb.XGBRegressor(
    objective="reg:squarederror",
    eval_metric=["rmse", "mae"],
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
)
```

| Parameter | Nilai | Keterangan |
|---|---|---|
| `objective` | `reg:squarederror` | Fungsi loss kuadrat untuk regresi |
| `eval_metric` | `rmse`, `mae` | Dicatat tiap iterasi untuk learning curve |
| `n_estimators` | 200 | Jumlah pohon |
| `learning_rate` | 0.05 | Ukuran langkah tiap pohon |
| `max_depth` | 6 | Kedalaman maksimum pohon |
| `subsample` | 0.8 | 80% baris dipakai per pohon |
| `colsample_bytree` | 0.8 | 80% fitur dipakai per pohon |
| `tree_method` | `hist` | Algoritma berbasis histogram (cepat) |
| `n_jobs` | -1 | Gunakan semua core CPU |

`eval_set` berisi data latih dan data uji, sehingga performa keduanya terekam di setiap iterasi.

### 3.7 Learning Curves

Dua grafik dari `model.evals_result()`:

- **RMSE per iterasi** (train vs test)
- **MAE per iterasi** (train vs test)

Gunakan grafik ini untuk mendeteksi *overfitting*: jika error train terus turun sementara error test mendatar atau naik, model terlalu menyesuaikan data latih.

### 3.8 Evaluasi

| Metrik | Rumus / Makna | Satuan |
|---|---|---|
| MAE | Rata-rata \|aktual − prediksi\| | °C |
| MSE | Rata-rata (aktual − prediksi)² | °C² |
| RMSE | √MSE, lebih sensitif terhadap error besar | °C |
| R² | Proporsi variansi target yang dijelaskan model (1 = sempurna) | – |

Visualisasi hasil: scatter **aktual vs prediksi** (2.000 sampel uji pertama) dengan garis ideal `y = x`. Titik yang rapat di sekitar garis menandakan prediksi akurat.

### 3.9 Tabel Sampel Prediksi

Tabel 10 baris data uji pertama dengan kolom:

| Kolom | Isi |
|---|---|
| `Waktu` | Timestamp observasi |
| `Suhu_Aktual_degC` | Nilai sebenarnya |
| `Prediksi_Model_degC` | Hasil prediksi model |
| `Selisih_Error_degC` | Prediksi − aktual |
| `Error_Absolut` | Nilai mutlak selisih |

---

## 4. Cara Menjalankan

1. Buka notebook di Google Colab, atau jalankan sebagai skrip Python lokal:
   ```bash
   python deteksi_suhu_cuaca_1_jam_kedepan_menggunakan_xgboost.py
   ```
2. Pastikan koneksi internet aktif (dataset diunduh dari Hugging Face).
3. Jika dijalankan sebagai skrip lokal (bukan notebook), grafik muncul satu per satu lewat `plt.show()`; tutup jendela grafik agar eksekusi berlanjut.

---

## 5. Catatan dan Saran Perbaikan

| # | Temuan | Saran |
|---|---|---|
| 1 | Hanya `wv (m/s)` yang dibersihkan dari nilai `-9999`. Dataset Jena juga memiliki kolom `max. wv (m/s)` dengan anomali serupa. | Terapkan penanganan yang sama pada `max. wv (m/s)`. |
| 2 | Data uji dipakai sebagai `eval_set` untuk learning curve, sehingga tidak ada set validasi terpisah. | Pisahkan menjadi train/validation/test, atau gunakan `TimeSeriesSplit`. |
| 3 | Tidak ada model pembanding (*baseline*). | Bandingkan dengan baseline persistence (`suhu_1_jam_lagi = suhu_sekarang`) untuk menilai nilai tambah model. |
| 4 | Kolom arah angin `wd (deg)` dipakai apa adanya, padahal sifatnya melingkar (0° = 360°). | Encode dengan sin/cos seperti jam dan bulan. |
| 5 | `shift` dan lag berbasis jumlah baris, bukan waktu. | Verifikasi tidak ada celah timestamp (interval harus konsisten 10 menit). |
| 6 | Hyperparameter ditetapkan manual. | Lakukan tuning (misalnya `RandomizedSearchCV` dengan `TimeSeriesSplit`) dan pertimbangkan `early_stopping_rounds`. |
| 7 | Importance fitur belum dianalisis. | Tambahkan `xgb.plot_importance(model)` untuk interpretasi. |

---

## 6. Struktur Output Skrip

1. Audit data mentah (dimensi, duplikat, missing, anomali sensor)
2. Perbandingan data sebelum dan sesudah preprocessing + statistik target
3. Grafik EDA (univariat, bivariat, multivariat)
4. Informasi ukuran data latih dan uji
5. Learning curves RMSE dan MAE
6. Metrik evaluasi MAE, MSE, RMSE, R²
7. Scatter aktual vs prediksi
8. Tabel 10 sampel hasil prediksi