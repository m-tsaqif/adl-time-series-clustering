# Ruang Sales — Analisis Clustering & Dashboard

Proyek analisis pola penjualan produk W0–W51 yang modular, reproducible, dan dilengkapi dashboard Streamlit lima halaman. Notebook menjadi sumber alur analisis; dashboard memakai fungsi visualisasi dan artefak hasil yang sama tanpa melatih ulang model ketika aplikasi dibuka.

## Struktur proyek

```text
Ruang_Sales_Analysis/
├── dataset/
│   └── sales.csv
├── notebook/
│   └── 01_analisis_clustering_fastdtw.ipynb
├── dashboard/
│   ├── 01_ringkasan.py
│   ├── 02_eksplorasi.py
│   ├── 03_segmentasi.py
│   ├── 04_perbandingan.py
│   └── 05_metodologi.py
├── scripts/
│   ├── data.py
│   ├── preprocessing.py
│   ├── distance.py
│   ├── clustering.py
│   ├── experiment.py
│   ├── visualization.py
│   ├── artifacts.py
│   ├── dashboard_data.py
│   └── dashboard_pages.py
├── assets/
│   └── style.css
├── .streamlit/
│   └── config.toml
├── model/
├── output/
│   ├── distance_cache/
│   └── plots/
├── config.py
├── app.py
├── validate_dashboard.py
└── requirements.txt
```

## Cara menjalankan

Disarankan Python 3.11 atau 3.12.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

python -m pip install -r requirements.txt
python -m ipykernel install --user --name ruang-sales --display-name "Python (Ruang Sales)"
```

Buka `notebook/01_analisis_clustering_fastdtw.ipynb`, pilih kernel `Python (Ruang Sales)`, lalu jalankan dari atas ke bawah. Notebook harus dijalankan dari root proyek atau folder `notebook`; path proyek sudah dicari otomatis di kedua lokasi tersebut.

Untuk dashboard, jalankan dari root proyek dengan `streamlit run app.py`. Dashboard memakai hasil analisis tersimpan dan tidak menjalankan ulang training saat aplikasi dibuka. Uji struktur dan konsistensi dasar dengan `python validate_dashboard.py`.

## Data dan preprocessing

Sumber data adalah `dataset/sales.csv` dengan struktur tepat `Product_Code, W0, W1, ..., W51`. Data yang dipakai untuk clustering selalu nilai penjualan asli pada kolom W0–W51. Tidak ada kolom `Normalized`, `min`, atau `max` yang dibaca sebagai fitur clustering.

Preprocessing menggunakan **Min-Max scaling per produk (row-wise)**. Setiap baris produk dipetakan ke rentang 0–1 berdasarkan nilai minimum dan maksimum produk itu sendiri. Pilihan ini sesuai dengan tujuan segmentasi bentuk pola mingguan: level volume dan besaran antarproduk dipisahkan dari kemiripan bentuk. Konsekuensinya, dua produk dengan volume sangat berbeda dapat mempunyai representasi pola yang serupa; karena itu volume, kontribusi, jumlah minggu nol, dan profil cluster selalu ditafsirkan kembali menggunakan data asli.

Catatan: `sklearn.preprocessing.MinMaxScaler` secara default melakukan scaling per kolom/fitur. Untuk time series berbentuk baris produk, `RowWiseMinMaxScaler` di `scripts/preprocessing.py` menerapkan rumus Min-Max di sepanjang 52 minggu pada masing-masing produk. Produk yang nilainya konstan dipetakan menjadi deret nol agar tidak terjadi pembagian nol.

## FastDTW dan enam metode

Notebook menghitung satu matriks jarak FastDTW yang dipakai bersama oleh enam metode: K-Means dengan DBA, K-Medoids, Agglomerative, DBSCAN, Fuzzy C-Means dengan DBA, dan Fuzzy C-Medoids. Matriks jarak disimpan dalam cache berbasis hash data dan radius sehingga tidak dihitung ulang bila input serta konfigurasi jarak tidak berubah.

Semua pipeline disimpan sebagai objek `sklearn.pipeline.Pipeline` yang berisi `RowWiseMinMaxScaler` dan estimator `FastDTWClusterer`. Kumpulan pipeline terpilih disimpan di `model/clustering_pipelines.joblib`; pipeline preprocessing disimpan terpisah. Estimator mempunyai `predict` untuk penempatan seri baru ke prototype terdekat. Untuk DBSCAN, seri baru ditempatkan ke cluster sampel inti terdekat jika jaraknya tidak melampaui `eps`, dan selain itu diberi label noise `-1`.

## Konfigurasi yang mudah diubah

Ubah parameter analisis di `config.py`, bukan di banyak cell notebook. Parameter utama:

- `FASTDTW_RADIUS`: radius FastDTW.
- `K_GRID`, `LINKAGE_GRID`: kandidat jumlah cluster dan linkage Agglomerative.
- `DBSCAN_MIN_SAMPLES_GRID`, `DBSCAN_EPS_QUANTILES`: pencarian parameter DBSCAN.
- `FUZZY_M_GRID`: derajat fuzziness untuk metode fuzzy.
- `KMEANS_MAX_ITER`, `FCM_MAX_ITER`, dan parameter iterasi lain: batas iterasi.
- `SILHOUETTE_TUNING_SAMPLE_SIZE`: jumlah sampel tetap untuk Silhouette saat menyaring kandidat; metrik model final dihitung kembali menggunakan seluruh anggota non-noise.
- `MIN_CLUSTER_SHARE`, `MIN_COVERAGE_PRIMARY`, `SHAPE_MAX_ETA_ZERO`: guardrail pemilihan hasil.
- `NUMBA_THREADS`: jumlah thread untuk matriks FastDTW.

Jumlah kombinasi kandidat dan iterasi memengaruhi waktu komputasi. Fuzzy C-Means memakai batas iterasi/DBA yang lebih konservatif karena setiap pusat fuzzy memperbarui alignment terhadap seluruh produk. Nilai iterasi tercatat pada output, sehingga pengguna dapat menaikkannya untuk analisis sensitivitas dengan konsekuensi waktu komputasi lebih besar. Mulai dari grid kecil ketika menguji perubahan kode.

## Evaluasi dan batas interpretasi

- **Silhouette FastDTW**: lebih tinggi menunjukkan pemisahan/kompaksi relatif lebih baik pada matriks jarak yang digunakan.
- **DBI medoid FastDTW**: indeks adaptasi berbasis medoid dan jarak FastDTW; lebih rendah lebih baik. Ini bukan DBI standar.
- **DBI dan CHI Euclidean**: dihitung pada fitur Min-Max untuk diagnostik tambahan, bukan pada geometri FastDTW.
- **ARI/NMI antar-metode**: mengukur kesamaan partisi, bukan akurasi atau ground truth.
- **Stabilitas ARI/NMI**: membandingkan fit ulang beberapa seed untuk metode yang menggunakan inisialisasi acak. Metode deterministik dicatat sebagai tidak diuji.
- **Coverage**: proporsi produk non-noise. Silhouette DBSCAN dihitung hanya untuk anggota non-noise sehingga harus dibaca bersama coverage.
- **PCA**: hanya proyeksi visual dua dimensi; PCA tidak digunakan untuk fitting clustering.

Skor multi-indeks adalah heuristik untuk membantu menyaring kandidat, bukan bukti bahwa satu metode benar secara universal. Seleksi utama juga memeriksa asosiasi label dengan jumlah minggu nol (`Eta2_Minggu_Nol`). Jika tidak ada metode dengan coverage penuh dan ukuran cluster minimum yang memenuhi `SHAPE_MAX_ETA_ZERO`, notebook memberi status `FALLBACK_NO_SHAPE_GUARDRAIL`: pemimpin skor tetap ditampilkan untuk referensi, tetapi tidak boleh disebut rekomendasi final atau segmentasi shape-only. Peringatan tambahan diberikan jika Silhouette sangat rendah atau stabilitas ARI rendah. Tidak ada label ground truth untuk mengukur akurasi segmentasi; hasil perlu dibaca sebagai analisis deskriptif/eksploratif.

## Output

Saat notebook dijalankan, hasil disimpan secara otomatis:

- `model/clustering_pipelines.joblib` — enam pipeline model terpilih.
- `model/preprocessing_pipeline.joblib` — pipeline Min-Max per produk.
- `output/product_labels.csv` — label dan keanggotaan fuzzy per produk.
- `output/metrics_comparison.csv`, `output/stability.csv`, `output/ari_matrix.csv`, `output/nmi_matrix.csv`.
- `output/tuning_*.csv`, `output/cluster_profiles_*.csv`, dan statistik mingguan/produk.
- `output/experiment_metadata.json` — konfigurasi, hash data, versi library, parameter hasil, status seleksi, dan peringatan kualitas.
- `output/plots/*.html` — visualisasi Plotly interaktif yang dibuat di `scripts/visualization.py`. Notebook dan dashboard memanggil fungsi visualisasi yang sama tanpa menulis ulang kode grafik.
- `output/distance_cache/*.npy` — cache matriks jarak untuk mempercepat eksekusi berikutnya.

## Dashboard Streamlit

Dashboard terdiri atas lima halaman terpisah: **Ringkasan**, **Eksplorasi Data**, **Segmentasi Produk**, **Perbandingan Metode**, dan **Metodologi**. Seluruh renderer antarmuka berada di `scripts/dashboard_pages.py`; helper pemuatan/validasi data ada di `scripts/dashboard_data.py`; definisi grafik tetap berada di `scripts/visualization.py`. Dengan demikian, notebook dan dashboard memakai fungsi grafik yang sama.

Halaman Segmentasi memungkinkan pemilihan metode, skala penjualan asli atau Min-Max, cluster, pencarian anggota, dan unduhan anggota. Halaman Perbandingan menyajikan metrik, coverage, ARI/NMI, tuning, Silhouette individual, serta proyeksi PCA. Filter tabel eksplorasi tidak mengubah label clustering. Saat aplikasi dibuka, dashboard memvalidasi hash data terhadap metadata, kecocokan label dan urutan produk, coverage, serta jumlah anggota profil agar artefak dari eksperimen berbeda tidak tercampur.

Jalankan secara lokal:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Untuk Windows, aktifkan virtual environment terlebih dahulu seperti instruksi di atas. Dashboard mempertahankan stylesheet dan tema asli dari proyek `Ruang_Sales_Streamlit` tanpa redesain: file `assets/style.css` dan `.streamlit/config.toml` disalin apa adanya. Komposisi warna hijau hutan–krem, aksen emas, font Fraunces/Plus Jakarta Sans, kartu, sidebar tombol, dan gaya toolbar Plotly dipertahankan; perubahan hanya pada integrasi artefak analisis dan pemisahan kode halaman. Saat mengunggah ke GitHub/Streamlit Community Cloud, sertakan folder `dataset/`, `model/`, `output/`, `scripts/`, `dashboard/`, `assets/`, `.streamlit/`, serta `config.py`, `app.py`, dan `requirements.txt`; artefak model dan output tidak dikecualikan oleh `.gitignore`.

## Catatan integrasi dengan dashboard sumber

Dashboard lama yang disertakan menggunakan skenario preprocessing `smooth5_log_zscore`. Hasil label, metrik, dan visualisasi dashboard lama tidak digunakan untuk dashboard baru karena analisis proyek ini mensyaratkan penjualan asli W0–W51 yang ditransformasikan dengan Min-Max per produk. Dashboard baru mengadopsi pola navigasi lima halaman dan tema visual hijau–krem, tetapi memuat seluruh tabel serta pipeline dari eksperimen Min-Max saat ini. Dengan demikian, tidak ada pencampuran label atau metrik dari dua preprocessing berbeda.
