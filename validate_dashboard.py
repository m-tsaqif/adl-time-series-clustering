"""Validasi statis dashboard, struktur artefak, dan konsistensi label data."""
from __future__ import annotations

import ast
import json
import hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent


def validate_project() -> list[str]:
    """Memeriksa keberadaan file dashboard dan kecocokan data/artefak yang digunakan."""
    errors: list[str] = []
    required = [
        "app.py", "config.py", "dashboard/01_ringkasan.py", "dashboard/02_eksplorasi.py",
        "dashboard/03_segmentasi.py", "dashboard/04_perbandingan.py", "dashboard/05_metodologi.py",
        "scripts/dashboard_data.py", "scripts/dashboard_pages.py", "assets/style.css",
        "dataset/sales.csv", "model/clustering_pipelines.joblib", "model/preprocessing_pipeline.joblib",
        "output/product_labels.csv", "output/metrics_comparison.csv", "output/experiment_metadata.json",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f"File wajib tidak ditemukan: {relative}")

    if errors:
        return errors

    sales = pd.read_csv(ROOT / "dataset/sales.csv")
    labels = pd.read_csv(ROOT / "output/product_labels.csv")
    metrics = pd.read_csv(ROOT / "output/metrics_comparison.csv")
    metadata = json.loads((ROOT / "output/experiment_metadata.json").read_text(encoding="utf-8"))
    dataset_hash = hashlib.sha256((ROOT / "dataset/sales.csv").read_bytes()).hexdigest()
    if metadata.get("source_sha256") and dataset_hash != metadata["source_sha256"]:
        errors.append("Hash dataset tidak sama dengan metadata eksperimen.")
    if sales.shape[0] != int(metadata["n_products"]):
        errors.append("Jumlah produk pada CSV tidak sesuai metadata.")
    if labels["Product_Code"].astype(str).tolist() != sales["Product_Code"].astype(str).tolist():
        errors.append("Urutan Product_Code label tidak sama dengan data penjualan.")
    if set(metrics["Metode"]) != set(metadata["methods"]):
        errors.append("Daftar metode di metrik tidak cocok dengan metadata.")
    profiles = pd.read_csv(ROOT / "output/cluster_profiles_all_methods.csv")
    for method in metadata["methods"]:
        if int(profiles.loc[profiles["Metode"] == method, "Jumlah_Produk"].sum()) != len(sales):
            errors.append(f"Jumlah anggota profil {method} tidak sama dengan jumlah produk.")
        actual_coverage = float((labels[method].to_numpy(dtype=int) != -1).mean())
        recorded_coverage = float(metrics.loc[metrics["Metode"] == method, "Coverage"].iloc[0])
        if abs(actual_coverage - recorded_coverage) > 1e-8:
            errors.append(f"Coverage label {method} tidak cocok dengan metrik.")
    for page in sorted((ROOT / "dashboard").glob("*.py")):
        ast.parse(page.read_text(encoding="utf-8"), filename=str(page))
    for script in [ROOT / "app.py", ROOT / "scripts/dashboard_data.py", ROOT / "scripts/dashboard_pages.py"]:
        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    return errors


if __name__ == "__main__":
    failures = validate_project()
    if failures:
        raise SystemExit("\n".join(failures))
    print("Dashboard validation passed: struktur, sintaks, label, metode, dan metadata konsisten.")
