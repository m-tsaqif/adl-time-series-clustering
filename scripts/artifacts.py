"""Penyimpanan hasil eksperimen, model pipeline, metadata, dan grafik interaktif."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import platform
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy
import sklearn
import numba
import plotly


def _sha256_file(path: Path) -> str:
    """Menghitung checksum SHA-256 sebuah file untuk pelacakan artefak."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_plotly_figures(
    figures: dict[str, go.Figure],
    output_dir: str | Path,
    include_plotlyjs: str = "cdn",
) -> list[Path]:
    """Menyimpan seluruh figure Plotly sebagai HTML interaktif dan mengembalikan path."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for name, figure in figures.items():
        destination = folder / f"{name}.html"
        figure.write_html(
            destination,
            include_plotlyjs=include_plotlyjs,
            full_html=True,
            config={"displaylogo": False, "scrollZoom": False, "responsive": True},
        )
        saved_paths.append(destination)
    return saved_paths


def save_experiment_artifacts(
    experiment: dict,
    data: pd.DataFrame,
    product_stats: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    profiles_by_method: dict[str, pd.DataFrame],
    silhouette_detail: pd.DataFrame,
    silhouette_summary: pd.DataFrame,
    pca_data: pd.DataFrame,
    config: object,
) -> dict[str, Path]:
    """Mengekspor pipeline, label, evaluasi, tuning, profil, dan metadata eksperimen."""
    model_dir = Path(config.MODEL_DIR)
    output_dir = Path(config.OUTPUT_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    export_tables = {
        "product_labels": experiment["labels"],
        "metrics_comparison": experiment["metrics"],
        "stability": experiment["stability"],
        "ari_matrix": experiment["ari"],
        "nmi_matrix": experiment["nmi"],
        "product_statistics": product_stats,
        "weekly_statistics": weekly_stats,
        "silhouette_per_product": silhouette_detail,
        "silhouette_per_cluster": silhouette_summary,
        "pca_coordinates": pca_data,
    }
    all_profiles = pd.concat(profiles_by_method.values(), ignore_index=True)
    export_tables["cluster_profiles_all_methods"] = all_profiles
    export_tables["tuning_all_methods"] = pd.concat(experiment["tuning"].values(), ignore_index=True)
    for name, table in export_tables.items():
        destination = output_dir / f"{name}.csv"
        table.to_csv(destination, index=(name in {"ari_matrix", "nmi_matrix"}), encoding="utf-8-sig")
        paths[name] = destination

    for method, table in experiment["tuning"].items():
        destination = output_dir / f"tuning_{method}.csv"
        table.to_csv(destination, index=False, encoding="utf-8-sig")
        paths[f"tuning_{method}"] = destination
    for method, table in profiles_by_method.items():
        destination = output_dir / f"cluster_profiles_{method}.csv"
        table.to_csv(destination, index=False, encoding="utf-8-sig")
        paths[f"cluster_profiles_{method}"] = destination

    pipeline_path = model_dir / "clustering_pipelines.joblib"
    joblib.dump(experiment["pipelines"], pipeline_path, compress=config.JOBLIB_COMPRESSION)
    paths["pipelines"] = pipeline_path
    preprocessing_path = model_dir / "preprocessing_pipeline.joblib"
    joblib.dump(experiment["preprocessing_pipeline"], preprocessing_path, compress=config.JOBLIB_COMPRESSION)
    paths["preprocessing_pipeline"] = preprocessing_path

    source_path = Path(config.DATA_FILE)
    data_digest = _sha256_file(source_path) if source_path.exists() else None
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source_path),
        "source_sha256": data_digest,
        "n_products": int(len(data)),
        "n_weeks": int(len(config.WEEK_COLUMNS)),
        "total_sales": float(data[config.WEEK_COLUMNS].to_numpy(dtype=float).sum()),
        "zero_observation_share": float((data[config.WEEK_COLUMNS].to_numpy(dtype=float) == 0).mean()),
        "preprocessing": "Row-wise MinMaxScaler on original W0-W51 for each product",
        "distance": {
            "name": "Numba multiresolution FastDTW",
            "radius": int(config.FASTDTW_RADIUS),
            "local_cost": config.FASTDTW_LOCAL_COST,
            "final_distance": "sqrt(sum of squared local costs on the alignment path)",
            "cache_file": str(experiment["distance_cache_path"]),
            "cache_hit": bool(experiment["distance_cache_hit"]),
        },
        "methods": list(experiment["pipelines"].keys()),
        "best_method": experiment["best_method"],
        "selection_status": experiment["selection_status"],
        "selection_warnings": experiment["selection_warnings"],
        "best_parameters": experiment["best_parameters"],
        "analysis_config": {
            "random_state": int(config.RANDOM_STATE),
            "minmax_axis": config.MINMAX_AXIS,
            "minmax_feature_range": list(config.MINMAX_FEATURE_RANGE),
            "k_grid": list(config.K_GRID),
            "linkage_grid": list(config.LINKAGE_GRID),
            "dbscan_min_samples_grid": list(config.DBSCAN_MIN_SAMPLES_GRID),
            "dbscan_eps_quantiles": list(config.DBSCAN_EPS_QUANTILES),
            "fuzzy_m_grid": list(config.FUZZY_M_GRID),
            "silhouette_tuning_sample_size": int(config.SILHOUETTE_TUNING_SAMPLE_SIZE),
            "min_cluster_share": float(config.MIN_CLUSTER_SHARE),
            "min_coverage_primary": float(config.MIN_COVERAGE_PRIMARY),
            "shape_max_eta_zero": float(config.SHAPE_MAX_ETA_ZERO),
            "stability_ari_min": float(config.STABILITY_ARI_MIN),
        },
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "numba": numba.__version__,
            "plotly": plotly.__version__, "joblib": joblib.__version__,
        },
        "notes": [
            "Clustering memakai W0-W51 asli yang ditransformasikan dengan Min-Max per produk.",
            "Volume, kontribusi, jumlah minggu nol, dan profil bisnis dihitung dari data asli.",
            "PCA hanya visualisasi dan tidak digunakan untuk menghitung cluster.",
            "Silhouette/DBI/CHI adalah indeks internal, bukan akurasi prediksi.",
            "DBI_Medoid_FastDTW merupakan adaptasi berbasis medoid, bukan DBI standar.",
            "DBSCAN dapat memberi label -1 untuk produk noise; coverage harus diperhatikan.",
            "Jika tidak ada model full-coverage yang lolos guardrail eta-squared minggu nol, pemimpin skor dilaporkan sebagai fallback, bukan rekomendasi final.",
        ],
    }
    metadata_path = output_dir / "experiment_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["metadata"] = metadata_path
    return paths
