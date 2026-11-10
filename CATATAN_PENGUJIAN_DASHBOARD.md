# Catatan Pengujian Dashboard

## Pemeriksaan yang telah dilakukan

- Sintaks `app.py`, semua halaman, dan modul dashboard lolos `compileall`.
- `python validate_dashboard.py` lulus untuk struktur file, urutan `Product_Code`, daftar metode, dan metadata eksperimen.
- Hash SHA-256 dataset sesuai dengan metadata eksperimen.
- Preprocessing pipeline berhasil dimuat dan menghasilkan matriks 811 × 52 dalam rentang Min-Max [0, 1].
- Jumlah anggota profil seluruh metode sesuai 811 produk; coverage label cocok dengan metrik yang diekspor.
- Seluruh 19 figure factory Plotly yang dipakai dashboard berhasil dibuat dari artefak terbaru.
- Lima renderer halaman berhasil melalui smoke test data/UI terisolasi.
- Paket unduhan artefak berhasil dibuat dan berisi dataset, label, metrik, pipeline, matriks jarak, serta grafik HTML.
- Audit AST memastikan setiap fungsi dan class di dalam folder `scripts/` memiliki docstring.

## Batas pengujian

Runtime Streamlit interaktif tidak dijalankan di lingkungan penyusunan karena paket `streamlit` tidak terpasang di lingkungan tersebut. Karena itu, pengujian di atas memverifikasi sintaks, data, fungsi grafik, dan alur renderer, bukan screenshot/browser test. Lakukan uji akhir di komputer lokal setelah memasang dependensi.

## Menjalankan uji lokal

```bash
python -m pip install -r requirements.txt
python validate_dashboard.py
python -m streamlit run app.py
```

Buka URL lokal yang ditampilkan Streamlit (umumnya `http://localhost:8501`).


## Pemeriksaan styling

- `assets/style.css` dibandingkan byte-per-byte dengan stylesheet proyek dashboard sumber; hasilnya identik.
- `.streamlit/config.toml` dibandingkan byte-per-byte dengan tema Streamlit sumber; hasilnya identik.
- Navigasi sidebar menggunakan tombol `st.button` dan selector `primary`/`secondary` seperti dashboard sumber, bukan tautan navigasi dengan aturan CSS tambahan.
- Gaya grafik Plotly mengikuti helper `plot()` sumber: font Plus Jakarta Sans, latar transparan, palet hijau/emas, margin, legend horizontal, dan mode bar saat hover.
- Fungsi visualisasi tetap berada di `scripts/visualization.py` agar dipakai bersama oleh notebook dan dashboard.
