"""Pemuatan, validasi, dan helper data khusus dashboard Ruang Sales."""
from __future__ import annotations

import io
import json
import zipfile
import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import silhouette_samples

import config
from scripts.data import load_sales_data, build_product_statistics, build_weekly_statistics


@st.cache_data(show_spinner="Memuat data dan artefak analisis...")
def load_dashboard_bundle() -> dict[str, Any]:
    """Memuat data asli, hasil clustering, metrik, pipeline, dan metadata secara konsisten."""
    sales = load_sales_data(
        config.DATA_FILE,
        id_column=config.ID_COLUMN,
        week_columns=config.WEEK_COLUMNS,
        delimiter=",",
        allow_missing=False,
        allow_negative=False,
    )
    weekly = sales[config.WEEK_COLUMNS].to_numpy(dtype=np.float64)
    product_codes = sales[config.ID_COLUMN].astype(str).to_numpy()

    preprocessing_path = config.MODEL_DIR / "preprocessing_pipeline.joblib"
    if not preprocessing_path.exists():
        raise FileNotFoundError(f"Pipeline preprocessing tidak ditemukan: {preprocessing_path}")
    preprocessing_pipeline = joblib.load(preprocessing_path)
    scaled = preprocessing_pipeline.transform(weekly)

    labels = pd.read_csv(config.OUTPUT_DIR / "product_labels.csv")
    if labels[config.ID_COLUMN].astype(str).tolist() != product_codes.tolist():
        raise ValueError("Urutan Product_Code pada product_labels.csv tidak cocok dengan dataset/sales.csv.")

    metrics = pd.read_csv(config.OUTPUT_DIR / "metrics_comparison.csv")
    methods = metrics["Metode"].astype(str).tolist()
    missing_methods = [method for method in methods if method not in labels.columns]
    if missing_methods:
        raise ValueError(f"Kolom label metode tidak lengkap: {missing_methods}")

    metadata = json.loads((config.OUTPUT_DIR / "experiment_metadata.json").read_text(encoding="utf-8"))
    current_hash = hashlib.sha256(config.DATA_FILE.read_bytes()).hexdigest()
    expected_hash = metadata.get("source_sha256")
    if expected_hash and current_hash != expected_hash:
        raise ValueError(
            "Hash dataset tidak sama dengan metadata eksperimen. "
            "Jalankan ulang notebook dan ekspor ulang seluruh artefak sebelum memakai dashboard."
        )
    if float(weekly.sum()) != float(metadata.get("total_sales", weekly.sum())):
        raise ValueError("Total penjualan dataset tidak sama dengan metadata eksperimen.")

    product_stats = build_product_statistics(sales, config.ID_COLUMN, config.WEEK_COLUMNS)
    weekly_stats = build_weekly_statistics(sales, config.WEEK_COLUMNS)

    profiles = pd.read_csv(config.OUTPUT_DIR / "cluster_profiles_all_methods.csv")
    tuning = pd.read_csv(config.OUTPUT_DIR / "tuning_all_methods.csv")
    ari = _read_similarity_matrix(config.OUTPUT_DIR / "ari_matrix.csv")
    nmi = _read_similarity_matrix(config.OUTPUT_DIR / "nmi_matrix.csv")
    silhouette_detail = pd.read_csv(config.OUTPUT_DIR / "silhouette_per_product.csv")
    silhouette_summary = pd.read_csv(config.OUTPUT_DIR / "silhouette_per_cluster.csv")
    pca_data = pd.read_csv(config.OUTPUT_DIR / "pca_coordinates.csv")

    expected_n, expected_w = len(sales), len(config.WEEK_COLUMNS)
    if weekly.shape != (expected_n, expected_w) or scaled.shape != weekly.shape:
        raise ValueError("Dimensi data asli dan hasil preprocessing tidak sesuai.")
    if not np.isfinite(scaled).all() or scaled.min() < -1e-9 or scaled.max() > 1 + 1e-9:
        raise ValueError("Hasil pipeline Min-Max tidak berada dalam rentang [0, 1].")
    for method in methods:
        method_profiles = profiles.loc[profiles["Metode"] == method]
        profile_total = int(method_profiles["Jumlah_Produk"].sum())
        if profile_total != expected_n:
            raise ValueError(f"Jumlah anggota profil {method} ({profile_total}) tidak sama dengan {expected_n} produk.")
        actual_coverage = float((labels[method].to_numpy(dtype=int) != -1).mean())
        recorded_coverage = float(metrics.loc[metrics["Metode"] == method, "Coverage"].iloc[0])
        if not np.isclose(actual_coverage, recorded_coverage, atol=1e-8):
            raise ValueError(f"Coverage label {method} tidak cocok dengan tabel metrik.")

    return {
        "sales": sales,
        "W": weekly,
        "X_scaled": scaled,
        "product_codes": product_codes,
        "product_stats": product_stats,
        "weekly_stats": weekly_stats,
        "labels": labels,
        "metrics": metrics,
        "profiles": profiles,
        "tuning": tuning,
        "ari": ari,
        "nmi": nmi,
        "silhouette_detail": silhouette_detail,
        "silhouette_summary": silhouette_summary,
        "pca_data": pca_data,
        "metadata": metadata,
        "methods": methods,
    }


def _read_similarity_matrix(path: Path) -> pd.DataFrame:
    """Membaca matriks ARI/NMI dengan nama metode sebagai indeks dan kolom."""
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    return matrix


def get_method_labels(bundle: dict[str, Any], method: str) -> np.ndarray:
    """Mengambil label cluster suatu metode sesuai urutan produk pada dataset."""
    if method not in bundle["labels"].columns:
        raise KeyError(f"Label untuk metode {method!r} tidak tersedia.")
    return bundle["labels"][method].to_numpy(dtype=int)


def get_method_profiles(bundle: dict[str, Any], method: str) -> pd.DataFrame:
    """Mengembalikan profil cluster untuk metode yang sedang dipilih."""
    return bundle["profiles"].loc[bundle["profiles"]["Metode"] == method].copy().sort_values("Cluster")


def get_method_tuning(bundle: dict[str, Any], method: str) -> pd.DataFrame:
    """Mengembalikan seluruh kandidat tuning untuk satu metode."""
    return bundle["tuning"].loc[bundle["tuning"]["Metode"] == method].copy()


def get_membership_columns(labels_frame: pd.DataFrame, method: str) -> list[str]:
    """Mencari kolom derajat keanggotaan fuzzy yang terkait dengan metode terpilih."""
    prefix = f"{method}_membership_"
    return [column for column in labels_frame.columns if column.startswith(prefix)]


@st.cache_data(show_spinner=False)
def calculate_silhouette_for_method(method: str, labels: tuple[int, ...]) -> pd.DataFrame:
    """Menghitung Silhouette individual dari matriks FastDTW untuk metode terpilih."""
    metadata_path = config.OUTPUT_DIR / "experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache_name = Path(metadata["distance"]["cache_file"]).name
    distance_path = config.DISTANCE_CACHE_DIR / cache_name
    if not distance_path.exists():
        return pd.DataFrame(columns=["Index", "Product_Code", "Cluster", "Silhouette"])

    distance = np.load(distance_path, mmap_mode="r")
    label_array = np.asarray(labels, dtype=int)
    mask = label_array != -1
    covered_labels = label_array[mask]
    if np.unique(covered_labels).size < 2 or len(covered_labels) <= np.unique(covered_labels).size:
        return pd.DataFrame(columns=["Index", "Product_Code", "Cluster", "Silhouette"])

    selected_distance = np.asarray(distance[np.ix_(mask, mask)], dtype=float)
    values = silhouette_samples(selected_distance, covered_labels, metric="precomputed")
    full_indices = np.flatnonzero(mask)
    return pd.DataFrame({
        "Index": full_indices,
        "Cluster": covered_labels,
        "Silhouette": values,
    })


def attach_product_codes_to_silhouette(detail: pd.DataFrame, product_codes: np.ndarray) -> pd.DataFrame:
    """Menambahkan kode produk pada detail Silhouette yang dihitung dari indeks baris."""
    frame = detail.copy()
    if not frame.empty:
        frame["Product_Code"] = np.asarray(product_codes)[frame["Index"].to_numpy(dtype=int)]
    return frame


@st.cache_data(show_spinner="Menyiapkan paket artefak...")
def build_results_zip() -> bytes:
    """Mengemas data, tabel hasil, metadata, matriks jarak, grafik, dan pipeline model."""
    buffer = io.BytesIO()
    included = [
        config.DATA_FILE,
        config.MODEL_DIR / "clustering_pipelines.joblib",
        config.MODEL_DIR / "preprocessing_pipeline.joblib",
        config.OUTPUT_DIR / "experiment_metadata.json",
    ]
    included.extend(sorted(config.OUTPUT_DIR.glob("*.csv")))
    included.extend(sorted(config.PLOT_DIR.glob("*.html")))
    metadata_path = config.OUTPUT_DIR / "experiment_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cache_name = Path(metadata.get("distance", {}).get("cache_file", "")).name
        if cache_name:
            included.append(config.DISTANCE_CACHE_DIR / cache_name)
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in included:
            if path.is_file():
                archive.write(path, arcname=str(path.relative_to(config.PROJECT_ROOT)))
    return buffer.getvalue()


def format_method_name(method: str) -> str:
    """Mengubah kode metode menjadi label yang lebih mudah dibaca di antarmuka."""
    names = {
        "KMeans_DBA": "K-Means + DBA",
        "KMedoids": "K-Medoids",
        "Agglomerative": "Agglomerative",
        "DBSCAN": "DBSCAN",
        "FuzzyCMeans": "Fuzzy C-Means + DBA",
        "FuzzyCMedoids": "Fuzzy C-Medoids",
    }
    return names.get(method, method)


def format_cluster_label(cluster: int) -> str:
    """Membentuk label tampilan untuk nomor cluster atau noise DBSCAN."""
    return "Noise (−1)" if int(cluster) == -1 else f"Cluster {int(cluster)}"
