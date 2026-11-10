"""Orkestrasi tuning, pemilihan model, evaluasi, dan profiling clustering."""
from __future__ import annotations

from itertools import combinations
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
    silhouette_samples, davies_bouldin_score, calinski_harabasz_score,
)
from sklearn.decomposition import PCA

from .clustering import METHODS, build_clustering_pipeline
from .distance import load_or_compute_distance_matrix
from .preprocessing import build_preprocessing_pipeline


def eta_squared_by_cluster(values: np.ndarray, labels: np.ndarray) -> float:
    """Mengukur proporsi variasi jumlah minggu nol yang terkait dengan label cluster."""
    x = np.asarray(values, dtype=float)
    y = np.asarray(labels)
    if x.size == 0 or len(x) != len(y):
        return np.nan
    denominator = float(np.sum((x - x.mean()) ** 2))
    if denominator <= 1e-12:
        return 0.0
    between = 0.0
    for cluster_id in np.unique(y):
        selected = x[y == cluster_id]
        if selected.size:
            between += selected.size * (selected.mean() - x.mean()) ** 2
    return float(between / denominator)


def calculate_medoid_dbi(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Menghitung indeks Davies–Bouldin adaptasi dengan medoid dan jarak FastDTW."""
    D = np.asarray(distance_matrix, dtype=float)
    y = np.asarray(labels, dtype=int)
    cluster_ids = sorted(set(y.tolist()) - {-1})
    if len(cluster_ids) < 2:
        return np.nan
    medoids, scatters = [], []
    for cluster_id in cluster_ids:
        members = np.flatnonzero(y == cluster_id)
        within = D[np.ix_(members, members)]
        medoid = int(members[np.argmin(within.sum(axis=1))])
        medoids.append(medoid)
        scatters.append(float(D[members, medoid].mean()))
    center_distances = D[np.ix_(medoids, medoids)].copy()
    np.fill_diagonal(center_distances, np.inf)
    scatter = np.asarray(scatters, dtype=float)
    ratios = (scatter[:, None] + scatter[None, :]) / np.maximum(center_distances, 1e-12)
    np.fill_diagonal(ratios, -np.inf)
    return float(np.mean(np.max(ratios, axis=1)))


def evaluate_labels(
    method: str,
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    X_scaled: np.ndarray,
    W_original: np.ndarray,
    params: dict | None = None,
    silhouette_sample_size: int | None = None,
    silhouette_random_state: int = 42,
) -> dict:
    """Menghitung kualitas cluster, coverage, ukuran cluster, dan diagnostik aktivitas."""
    y = np.asarray(labels, dtype=int)
    covered = y != -1
    cluster_ids = np.unique(y[covered])
    n_covered = int(covered.sum())
    n_total = len(y)
    counts = np.array([(y == cluster_id).sum() for cluster_id in cluster_ids], dtype=int)
    record = {
        "Metode": method,
        "params": json.dumps(params or {}, ensure_ascii=False, sort_keys=True),
        "Jumlah_Cluster": int(len(cluster_ids)),
        "Jumlah_Noise": int((~covered).sum()),
        "Coverage": float(n_covered / n_total),
        "Anggota_Min": int(counts.min()) if counts.size else 0,
        "Anggota_Max": int(counts.max()) if counts.size else 0,
        "Silhouette_FastDTW": np.nan,
        "DBI_Medoid_FastDTW": np.nan,
        "DBI_MinMax_Euclidean": np.nan,
        "CHI_MinMax_Euclidean": np.nan,
        "Eta2_Minggu_Nol": np.nan,
    }
    if n_covered:
        zeros = (np.asarray(W_original)[covered] == 0).sum(axis=1)
        record["Eta2_Minggu_Nol"] = eta_squared_by_cluster(zeros, y[covered])
    if len(cluster_ids) < 2 or len(cluster_ids) >= n_covered:
        return record

    kept = np.flatnonzero(covered)
    y_kept = y[covered]
    d_kept = np.asarray(distance_matrix)[np.ix_(kept, kept)]
    x_kept = np.asarray(X_scaled)[covered]
    effective_sample_size = (
        min(int(silhouette_sample_size), n_covered)
        if silhouette_sample_size is not None else None
    )
    record["Silhouette_FastDTW"] = float(silhouette_score(
        d_kept, y_kept, metric="precomputed", sample_size=effective_sample_size,
        random_state=silhouette_random_state,
    ))
    record["DBI_Medoid_FastDTW"] = calculate_medoid_dbi(distance_matrix, y)
    record["DBI_MinMax_Euclidean"] = float(davies_bouldin_score(x_kept, y_kept))
    record["CHI_MinMax_Euclidean"] = float(calinski_harabasz_score(x_kept, y_kept))
    return record


def _candidate_specs(config: object) -> dict[str, list[dict]]:
    """Menyusun kombinasi hyperparameter enam metode berdasarkan konfigurasi proyek."""
    candidates: dict[str, list[dict]] = {
        "KMeans_DBA": [], "KMedoids": [], "Agglomerative": [],
        "DBSCAN": [], "FuzzyCMeans": [], "FuzzyCMedoids": [],
    }
    for k in config.K_GRID:
        candidates["KMeans_DBA"].append({"n_clusters": int(k)})
        candidates["KMedoids"].append({"n_clusters": int(k)})
        for linkage_method in config.LINKAGE_GRID:
            candidates["Agglomerative"].append({"n_clusters": int(k), "linkage_method": linkage_method})
        for fuzziness in config.FUZZY_M_GRID:
            candidates["FuzzyCMeans"].append({"n_clusters": int(k), "fuzziness": float(fuzziness)})
            candidates["FuzzyCMedoids"].append({"n_clusters": int(k), "fuzziness": float(fuzziness)})
    return candidates


def _rank_candidate_table(table: pd.DataFrame, n_products: int, config: object) -> pd.DataFrame:
    """Memberi ranking komposit pada kandidat yang lolos guardrail coverage dan ukuran."""
    ranked = table.copy()
    minimum_size = int(np.ceil(n_products * config.MIN_CLUSTER_SHARE))
    valid = (
        ranked["Jumlah_Cluster"].between(2, 12)
        & ranked["Silhouette_FastDTW"].notna()
        & (ranked["Coverage"] >= 1.0 - config.MAX_NOISE_SHARE_FOR_TUNING)
        & (ranked["Anggota_Min"] >= minimum_size)
    )
    ranked["Tersaring"] = valid
    score = pd.Series(0.0, index=ranked.index)
    for metric, (weight, higher_is_better) in config.METRIC_WEIGHTS.items():
        values = ranked.loc[valid, metric]
        if values.empty:
            continue
        percentile = values.rank(ascending=not higher_is_better, pct=True, method="average")
        score.loc[valid] += float(weight) * percentile
    ranked["Skor_Multi_Indeks"] = np.where(valid, score, np.nan)
    return ranked


def _select_candidate(table: pd.DataFrame, n_products: int, config: object) -> tuple[pd.DataFrame, int]:
    """Memilih satu konfigurasi per metode dan menandai fallback bila guardrail gagal."""
    ranked = _rank_candidate_table(table, n_products, config)
    feasible = ranked.dropna(subset=["Skor_Multi_Indeks"])
    if not feasible.empty:
        selected_index = feasible.sort_values(
            ["Skor_Multi_Indeks", "Silhouette_FastDTW"], ascending=False
        ).index[0]
    else:
        fallback = ranked[ranked["Jumlah_Cluster"] >= 2]
        if fallback.empty:
            selected_index = ranked.sort_values(["Coverage", "Jumlah_Cluster"], ascending=False).index[0]
        else:
            selected_index = fallback.sort_values(
                ["Anggota_Min", "Coverage", "Silhouette_FastDTW"], ascending=False
            ).index[0]
    return ranked, int(selected_index)


def _pipeline_params(method: str, params: dict, config: object) -> dict:
    """Menggabungkan parameter kandidat dengan pengaturan algoritma pada config.py."""
    common = {
        "radius": config.FASTDTW_RADIUS,
        "max_iter": config.KMEANS_MAX_ITER,
        "n_init": config.KMEANS_N_INIT,
        "dba_iterations": config.DBA_ITERATIONS,
        "tolerance": config.FCM_TOLERANCE,
    }
    if method == "KMedoids":
        common.update(max_iter=config.KMEDOIDS_MAX_ITER, n_init=config.KMEDOIDS_N_INIT)
    elif method == "FuzzyCMeans":
        common.update(max_iter=config.FCM_MAX_ITER, dba_iterations=config.FCM_DBA_ITERATIONS)
    elif method == "FuzzyCMedoids":
        common.update(max_iter=config.FCMEDOIDS_MAX_ITER)
    elif method in {"Agglomerative", "DBSCAN"}:
        common.update(max_iter=1, n_init=1)
    common.update(params)
    return common


def _stability_for_pipeline(
    method: str,
    params: dict,
    W: np.ndarray,
    D: np.ndarray,
    config: object,
) -> dict:
    """Mengukur konsistensi hasil fit ulang melalui ARI/NMI untuk metode stokastik."""
    stochastic_methods = {"KMeans_DBA", "KMedoids", "FuzzyCMeans", "FuzzyCMedoids"}
    if method not in stochastic_methods:
        return {
            "Metode": method, "Stabilitas_ARI": np.nan, "Stabilitas_NMI": np.nan,
            "Status": "Deterministik; uji seed tidak diperlukan",
        }
    runs = []
    for seed in config.STABILITY_SEEDS:
        pipeline = build_clustering_pipeline(
            method=method, distance_matrix=D,
            random_state=int(seed), **_pipeline_params(method, params, config),
        )
        pipeline.fit(W)
        runs.append(pipeline.named_steps["cluster"].labels_.copy())
    pairs = list(combinations(range(len(runs)), 2))
    ari_values = [adjusted_rand_score(runs[i], runs[j]) for i, j in pairs]
    nmi_values = [normalized_mutual_info_score(runs[i], runs[j]) for i, j in pairs]
    return {
        "Metode": method,
        "Stabilitas_ARI": float(np.mean(ari_values)) if ari_values else np.nan,
        "Stabilitas_NMI": float(np.mean(nmi_values)) if nmi_values else np.nan,
        "Status": f"Uji {len(config.STABILITY_SEEDS)} seed",
    }


def run_clustering_experiment(W_original: np.ndarray, product_codes: np.ndarray, config: object) -> dict:
    """Menjalankan preprocessing, matriks FastDTW, tuning enam metode, evaluasi, dan seleksi."""
    W = np.asarray(W_original, dtype=np.float64)
    ids = np.asarray(product_codes, dtype=str)
    if W.ndim != 2 or len(W) != len(ids) or not np.isfinite(W).all():
        raise ValueError("W_original harus finite dan sebaris dengan product_codes.")
    if W.shape[1] != config.EXPECTED_N_WEEKS:
        raise ValueError(f"Jumlah minggu harus {config.EXPECTED_N_WEEKS}.")

    preprocessing_pipeline = build_preprocessing_pipeline(config.MINMAX_FEATURE_RANGE)
    X_scaled = preprocessing_pipeline.fit_transform(W)
    D, distance_path, cache_hit = load_or_compute_distance_matrix(
        X_scaled, config.FASTDTW_RADIUS, config.DISTANCE_CACHE_DIR,
        reuse_cache=config.REUSE_DISTANCE_CACHE, n_threads=config.NUMBA_THREADS,
    )

    candidates_by_method = _candidate_specs(config)
    tuning_tables: dict[str, pd.DataFrame] = {}
    best_pipelines: dict[str, object] = {}
    best_parameters: dict[str, dict] = {}
    runtime_by_method: dict[str, float] = {}

    sorted_distances = np.sort(D, axis=1)
    dbscan_specs = []
    for min_samples in config.DBSCAN_MIN_SAMPLES_GRID:
        neighbor_index = min(int(min_samples), len(D) - 1)
        k_distances = sorted_distances[:, neighbor_index]
        for quantile in config.DBSCAN_EPS_QUANTILES:
            eps = max(float(np.quantile(k_distances, quantile)), 1e-8)
            dbscan_specs.append({
                "min_samples": int(min_samples), "eps": eps,
                "eps_quantile": float(quantile),
            })
    candidates_by_method["DBSCAN"] = dbscan_specs

    for method in METHODS:
        records = []
        candidate_pipelines = []
        started = time.perf_counter()
        if getattr(config, "VERBOSE_EXPERIMENT", False):
            print(f"[{method}] {len(candidates_by_method[method])} kandidat...", flush=True)
        for params in candidates_by_method[method]:
            candidate_started = time.perf_counter()
            estimator_params = {key: value for key, value in params.items() if key != "eps_quantile"}
            model_params = _pipeline_params(method, estimator_params, config)
            pipeline = build_clustering_pipeline(
                method=method, distance_matrix=D,
                random_state=config.RANDOM_STATE, **model_params,
            )
            pipeline.fit(W)
            clusterer = pipeline.named_steps["cluster"]
            row = evaluate_labels(
                method, clusterer.labels_, D, X_scaled, W, params=params,
                silhouette_sample_size=config.SILHOUETTE_TUNING_SAMPLE_SIZE,
                silhouette_random_state=config.RANDOM_STATE,
            )
            row["Runtime_Algoritma_s"] = float(time.perf_counter() - candidate_started)
            row["Iterasi"] = int(clusterer.n_iter_)
            row["Objektif"] = float(clusterer.objective_) if np.isfinite(clusterer.objective_) else np.nan
            if hasattr(clusterer, "fpc_"):
                row["FPC"] = clusterer.fpc_
                row["Partition_Entropy"] = clusterer.partition_entropy_
            if method == "DBSCAN":
                row.update(params)
            if method == "Agglomerative":
                row["Linkage"] = params["linkage_method"]
            records.append(row)
            candidate_pipelines.append((pipeline, params.copy()))

        table = pd.DataFrame(records)
        ranked, best_index = _select_candidate(table, len(W), config)
        best_pipelines[method] = candidate_pipelines[best_index][0]
        best_parameters[method] = candidate_pipelines[best_index][1]
        tuning_tables[method] = ranked
        runtime_by_method[method] = float(time.perf_counter() - started)
        if getattr(config, "VERBOSE_EXPERIMENT", False):
            print(f"[{method}] selesai dalam {runtime_by_method[method]:.1f} detik.", flush=True)

    stability_records = []
    for method in METHODS:
        if getattr(config, "VERBOSE_EXPERIMENT", False):
            print(f"[Stabilitas] {method}...", flush=True)
        result = _stability_for_pipeline(
            method, best_parameters[method], W, D, config,
        )
        stability_records.append(result)
    stability_table = pd.DataFrame(stability_records)

    metric_rows = []
    label_data = {"Product_Code": ids}
    for method in METHODS:
        pipeline = best_pipelines[method]
        clusterer = pipeline.named_steps["cluster"]
        labels = clusterer.labels_
        label_data[method] = labels
        if clusterer.membership_ is not None:
            for cluster_index in range(clusterer.membership_.shape[1]):
                label_data[f"{method}_membership_C{cluster_index}"] = clusterer.membership_[:, cluster_index]
            label_data[f"{method}_Confidence"] = clusterer.membership_.max(axis=1)
        selected_params = best_parameters[method]
        evaluation = evaluate_labels(method, labels, D, X_scaled, W, params=selected_params)
        evaluation["Runtime_Tuning_s"] = runtime_by_method[method]
        evaluation["Iterasi"] = int(clusterer.n_iter_)
        if hasattr(clusterer, "fpc_"):
            evaluation["FPC"] = clusterer.fpc_
            evaluation["Partition_Entropy"] = clusterer.partition_entropy_
        metric_rows.append(evaluation)
    metrics = pd.DataFrame(metric_rows).merge(stability_table, on="Metode", how="left")
    metrics["Layak_Semua_Produk"] = metrics["Coverage"] >= config.MIN_COVERAGE_PRIMARY - 1e-10

    # Skor lintas metode dihitung hanya untuk kandidat yang cakupan dan ukuran clusternya layak.
    minimum_size = int(np.ceil(len(W) * config.MIN_CLUSTER_SHARE))
    eligible_mask = (
        metrics["Layak_Semua_Produk"]
        & (metrics["Jumlah_Cluster"] >= 2)
        & (metrics["Anggota_Min"] >= minimum_size)
        & metrics["Silhouette_FastDTW"].notna()
    )
    eligible = metrics.loc[eligible_mask].copy()
    metrics["Skor_Multi_Indeks"] = np.nan
    if eligible.empty:
        raise RuntimeError("Tidak ada metode yang memenuhi syarat segmentasi penuh dan ukuran minimum.")

    # Guardrail shape diterapkan secara eksplisit. Jika tidak ada model yang lolos,
    # hasil tetap diranking sebagai fallback dan tidak disebut rekomendasi final.
    shape_eligible = eligible[
        eligible["Eta2_Minggu_Nol"] <= config.SHAPE_MAX_ETA_ZERO
    ].copy()
    if not shape_eligible.empty:
        ranking_pool = shape_eligible
        selection_status = "PASS_SHAPE_GUARDRAIL"
        selection_warnings = []
    else:
        ranking_pool = eligible
        selection_status = "FALLBACK_NO_SHAPE_GUARDRAIL"
        selection_warnings = [
            (
                "Tidak ada metode dengan coverage penuh dan ukuran cluster minimum "
                f"yang memenuhi eta-squared minggu nol <= {config.SHAPE_MAX_ETA_ZERO:.2f}. "
                "Ranking yang ditampilkan hanya pemimpin skor multi-indeks sementara; "
                "hasil belum layak disebut segmentasi shape-only."
            )
        ]

    scores = pd.Series(0.0, index=ranking_pool.index)
    for metric, (weight, higher_is_better) in config.METRIC_WEIGHTS.items():
        vals = ranking_pool[metric]
        percentiles = vals.rank(ascending=not higher_is_better, pct=True, method="average")
        scores += float(weight) * percentiles
    metrics.loc[ranking_pool.index, "Skor_Multi_Indeks"] = scores
    best_method = str(metrics.loc[ranking_pool.index].sort_values(
        ["Skor_Multi_Indeks", "Silhouette_FastDTW"], ascending=False
    ).iloc[0]["Metode"])

    best_row = metrics.loc[metrics["Metode"] == best_method].iloc[0]
    if pd.notna(best_row["Silhouette_FastDTW"]) and best_row["Silhouette_FastDTW"] < 0.10:
        selection_warnings.append(
            "Silhouette FastDTW pemimpin ranking di bawah 0.10; pemisahan cluster relatif lemah."
        )
    if pd.notna(best_row["Stabilitas_ARI"]) and best_row["Stabilitas_ARI"] < config.STABILITY_ARI_MIN:
        selection_warnings.append(
            "Stabilitas ARI pemimpin ranking di bawah ambang konfigurasi "
            f"({config.STABILITY_ARI_MIN:.2f}); hasil sensitif terhadap inisialisasi."
        )
    if not shape_eligible.empty:
        selection_warnings.append(
            "Lolos guardrail eta-squared bukan bukti validitas eksternal; interpretasi bisnis tetap perlu divalidasi."
        )

    labels_frame = pd.DataFrame(label_data)
    labels_frame["Total_Sales_52Minggu"] = W.sum(axis=1)
    labels_frame["Jumlah_Minggu_Nol"] = (W == 0).sum(axis=1)
    labels_frame["Proporsi_Minggu_Aktif"] = (W > 0).mean(axis=1)

    method_names = list(METHODS)
    ari = pd.DataFrame(
        [[adjusted_rand_score(labels_frame[a], labels_frame[b]) for b in method_names] for a in method_names],
        index=method_names, columns=method_names,
    )
    nmi = pd.DataFrame(
        [[normalized_mutual_info_score(labels_frame[a], labels_frame[b]) for b in method_names] for a in method_names],
        index=method_names, columns=method_names,
    )

    return {
        "preprocessing_pipeline": preprocessing_pipeline,
        "X_scaled": X_scaled,
        "D": D,
        "distance_cache_path": str(distance_path),
        "distance_cache_hit": bool(cache_hit),
        "pipelines": best_pipelines,
        "best_parameters": best_parameters,
        "tuning": tuning_tables,
        "metrics": metrics.sort_values("Skor_Multi_Indeks", ascending=False, na_position="last").reset_index(drop=True),
        "stability": stability_table,
        "labels": labels_frame,
        "ari": ari,
        "nmi": nmi,
        "best_method": best_method,
        "selection_status": selection_status,
        "selection_warnings": selection_warnings,
        "runtime_by_method": runtime_by_method,
    }


def build_cluster_profiles(
    W_original: np.ndarray,
    product_codes: np.ndarray,
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    method: str,
) -> pd.DataFrame:
    """Membentuk profil bisnis cluster menggunakan penjualan asli dan medoid FastDTW."""
    W = np.asarray(W_original, dtype=float)
    ids = np.asarray(product_codes, dtype=str)
    y = np.asarray(labels, dtype=int)
    rows = []
    total_all = float(W.sum())
    for cluster_id in sorted(np.unique(y)):
        members = np.flatnonzero(y == cluster_id)
        group = W[members]
        product_totals = group.sum(axis=1)
        zero_counts = (group == 0).sum(axis=1)
        if cluster_id == -1:
            medoid_code = "—"
            medoid_index = -1
        else:
            within = distance_matrix[np.ix_(members, members)]
            medoid_position = int(np.argmin(within.sum(axis=1)))
            medoid_index = int(members[medoid_position])
            medoid_code = str(ids[medoid_index])
        first_half = float(group[:, :26].mean())
        second_half = float(group[:, 26:].mean())
        rows.append({
            "Metode": method,
            "Cluster": int(cluster_id),
            "Jumlah_Produk": int(len(members)),
            "Persen_Produk": float(100.0 * len(members) / len(W)),
            "Total_Volume": float(group.sum()),
            "Porsi_Volume": float(100.0 * group.sum() / total_all) if total_all else 0.0,
            "Median_Total_Sales": float(np.median(product_totals)),
            "Rata2_Total_Sales": float(np.mean(product_totals)),
            "Median_Minggu_Nol": float(np.median(zero_counts)),
            "Rata2_Minggu_Nol": float(np.mean(zero_counts)),
            "Persen_Minggu_Aktif": float(100.0 * (group > 0).mean()),
            "Rerata_Penjualan_Paruh_1": first_half,
            "Rerata_Penjualan_Paruh_2": second_half,
            "Perubahan_Paruh_2_vs_1_Persen": float(100.0 * (second_half - first_half) / first_half) if first_half else np.nan,
            "Produk_Medoid": medoid_code,
            "Medoid_Index": medoid_index,
        })
    return pd.DataFrame(rows)


def build_silhouette_detail(labels: np.ndarray, distance_matrix: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Menghasilkan skor silhouette per produk dan ringkasan diagnostik per cluster."""
    y = np.asarray(labels, dtype=int)
    keep = y != -1
    if len(np.unique(y[keep])) < 2 or int(keep.sum()) <= len(np.unique(y[keep])):
        return pd.DataFrame(columns=["Index", "Cluster", "Silhouette"]), pd.DataFrame()
    selected_indices = np.flatnonzero(keep)
    scores = silhouette_samples(distance_matrix[np.ix_(selected_indices, selected_indices)], y[keep], metric="precomputed")
    detail = pd.DataFrame({"Index": selected_indices, "Cluster": y[keep], "Silhouette": scores})
    summary = detail.groupby("Cluster").agg(
        Jumlah_Produk=("Silhouette", "size"),
        Rata_Silhouette=("Silhouette", "mean"),
        Median_Silhouette=("Silhouette", "median"),
        Persen_Silhouette_Negatif=("Silhouette", lambda values: 100.0 * (values < 0).mean()),
    ).reset_index()
    return detail, summary


def build_tuning_per_k(tuning_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Memilih konfigurasi terbaik untuk setiap jumlah cluster aktual dari tabel tuning."""
    rows = []
    for method, table in tuning_tables.items():
        if method == "DBSCAN" or table.empty:
            continue
        valid_table = table[table["Jumlah_Cluster"] >= 2]
        for actual_k, subset in valid_table.groupby("Jumlah_Cluster"):
            feasible = subset.dropna(subset=["Skor_Multi_Indeks"])
            selected = (feasible if not feasible.empty else subset).sort_values(
                ["Skor_Multi_Indeks", "Silhouette_FastDTW"], ascending=False, na_position="last"
            ).iloc[0]
            record = selected.to_dict()
            record["Metode"] = method
            record["k_Aktual"] = int(actual_k)
            record["Lolos_Guardrail"] = bool(pd.notna(selected.get("Skor_Multi_Indeks", np.nan)))
            rows.append(record)
    return pd.DataFrame(rows)


def build_pca_coordinates(X_scaled: np.ndarray, labels: np.ndarray, random_state: int = 42) -> pd.DataFrame:
    """Membuat proyeksi PCA dua dimensi untuk eksplorasi visual, bukan sebagai jarak clustering."""
    coords = PCA(n_components=2, random_state=random_state).fit_transform(X_scaled)
    return pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Cluster": np.asarray(labels, dtype=int)})
