"""Estimator clustering enam metode berbasis matriks jarak FastDTW."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline

from .distance import compute_distance_matrix, compute_cross_distance, fastdtw_numba
from .preprocessing import RowWiseMinMaxScaler

METHODS = (
    "KMeans_DBA", "KMedoids", "Agglomerative", "DBSCAN", "FuzzyCMeans", "FuzzyCMedoids"
)


def _init_medoids_pp(distance: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
    """Menginisialisasi prototype dengan strategi k-means++ berbasis jarak."""
    rng = np.random.default_rng(random_state)
    n_samples = distance.shape[0]
    chosen = [int(rng.integers(n_samples))]
    while len(chosen) < n_clusters:
        nearest = np.min(distance[:, chosen], axis=1)
        weights = nearest ** 2
        weights[chosen] = 0.0
        if weights.sum() > 0:
            candidate = int(rng.choice(n_samples, p=weights / weights.sum()))
        else:
            unused = np.setdiff1d(np.arange(n_samples), np.asarray(chosen))
            candidate = int(rng.choice(unused))
        chosen.append(candidate)
    return np.asarray(chosen, dtype=int)


def _weighted_dba(
    X: np.ndarray,
    weights: np.ndarray,
    initial: np.ndarray,
    radius: int,
    n_iterations: int,
) -> np.ndarray:
    """Memperbarui satu barycenter dengan Dynamic Time Warping Barycenter Averaging."""
    center = np.asarray(initial, dtype=np.float64).copy()
    weight_values = np.asarray(weights, dtype=np.float64)
    for _ in range(int(n_iterations)):
        sums = np.zeros(len(center), dtype=np.float64)
        counts = np.zeros(len(center), dtype=np.float64)
        for sample_index, series in enumerate(X):
            weight = weight_values[sample_index]
            if weight <= 1e-12:
                continue
            _, path_i, path_j = fastdtw_numba(center, series, int(radius), True)
            for i, j in zip(path_i, path_j):
                sums[i] += weight * series[j]
                counts[i] += weight
        valid = counts > 1e-12
        center[valid] = sums[valid] / counts[valid]
    return center


def memberships_from_distances(distances: np.ndarray, fuzziness: float) -> np.ndarray:
    """Menghitung matriks keanggotaan fuzzy dari jarak ke setiap pusat."""
    distance = np.asarray(distances, dtype=float)
    if fuzziness <= 1:
        raise ValueError("Parameter fuzziness harus lebih besar dari 1.")
    power = 2.0 / (fuzziness - 1.0)
    membership = np.zeros_like(distance, dtype=float)
    zero_mask = distance <= 1e-12
    zero_rows = zero_mask.any(axis=1)
    if zero_rows.any():
        membership[zero_rows] = zero_mask[zero_rows] / zero_mask[zero_rows].sum(axis=1, keepdims=True)
    if (~zero_rows).any():
        inverse = np.maximum(distance[~zero_rows], 1e-12) ** (-power)
        membership[~zero_rows] = inverse / inverse.sum(axis=1, keepdims=True)
    return membership


def _medoid_indices(distance: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mengambil medoid untuk setiap cluster dan mengembalikan ID cluster terkait."""
    medoids, cluster_ids = [], []
    for cluster_id in sorted(set(np.asarray(labels).tolist()) - {-1}):
        members = np.flatnonzero(labels == cluster_id)
        if members.size == 0:
            continue
        within = distance[np.ix_(members, members)]
        medoids.append(int(members[np.argmin(within.sum(axis=1))]))
        cluster_ids.append(int(cluster_id))
    return np.asarray(medoids, dtype=int), np.asarray(cluster_ids, dtype=int)


class FastDTWClusterer(BaseEstimator, ClusterMixin):
    """Estimator enam metode clustering time series dengan FastDTW sebagai jarak utama."""

    def __init__(
        self,
        method: str = "KMedoids",
        n_clusters: int = 2,
        linkage_method: str = "average",
        eps: float = 1.0,
        min_samples: int = 7,
        fuzziness: float = 1.7,
        radius: int = 2,
        random_state: int = 42,
        max_iter: int = 30,
        n_init: int = 1,
        dba_iterations: int = 5,
        tolerance: float = 1e-4,
        distance_matrix: np.ndarray | None = None,
    ):
        """Menginisialisasi metode, hyperparameter, dan matriks jarak opsional."""
        self.method = method
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.eps = eps
        self.min_samples = min_samples
        self.fuzziness = fuzziness
        self.radius = radius
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.dba_iterations = dba_iterations
        self.tolerance = tolerance
        self.distance_matrix = distance_matrix

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "FastDTWClusterer":
        """Melatih metode terpilih; matriks jarak dapat dipakai bersama agar tidak dihitung ulang."""
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("X harus berupa matriks dua dimensi tanpa NaN/inf.")
        if self.method not in METHODS:
            raise ValueError(f"Metode tidak dikenal: {self.method}. Pilihan: {METHODS}")
        if self.n_clusters < 2 and self.method != "DBSCAN":
            raise ValueError("n_clusters minimal 2.")
        if self.radius < 1:
            raise ValueError("radius FastDTW minimal 1.")
        self.n_features_in_ = values.shape[1]
        self.X_fit_ = np.ascontiguousarray(values, dtype=np.float64)
        if self.distance_matrix is None:
            distance = compute_distance_matrix(self.X_fit_, radius=self.radius)
        else:
            distance = np.asarray(self.distance_matrix, dtype=np.float64)
        expected = (len(values), len(values))
        if distance.shape != expected or not np.isfinite(distance).all():
            raise ValueError(f"Matriks jarak harus berukuran {expected} dan bernilai finite.")
        if not np.allclose(distance, distance.T, atol=1e-8) or not np.allclose(np.diag(distance), 0.0):
            raise ValueError("Matriks jarak harus simetris dan diagonalnya nol.")
        if (distance < 0).any():
            raise ValueError("Matriks jarak tidak boleh negatif.")
        self.distance_matrix_ = distance
        self.membership_ = None
        self.cluster_centers_ = None
        self.medoid_indices_ = None
        self.core_sample_indices_ = None
        self.objective_ = np.nan
        self.n_iter_ = 0

        if self.method == "KMeans_DBA":
            self._fit_kmeans_dba()
        elif self.method == "KMedoids":
            self._fit_kmedoids()
        elif self.method == "Agglomerative":
            self._fit_agglomerative()
        elif self.method == "DBSCAN":
            self._fit_dbscan()
        elif self.method == "FuzzyCMeans":
            self._fit_fuzzy_cmeans()
        elif self.method == "FuzzyCMedoids":
            self._fit_fuzzy_cmedoids()

        self.labels_ = np.asarray(self.labels_, dtype=int)
        self.classes_ = np.unique(self.labels_)
        self.n_clusters_ = len(set(self.labels_.tolist()) - {-1})
        self.cluster_medoid_indices_, self.cluster_medoid_labels_ = _medoid_indices(
            self.distance_matrix_, self.labels_
        )
        if self.method == "Agglomerative":
            self.medoid_indices_ = self.cluster_medoid_indices_.copy()
            self.prototype_cluster_labels_ = self.cluster_medoid_labels_.copy()
        elif self.method in {"KMedoids", "FuzzyCMedoids"}:
            self.prototype_cluster_labels_ = np.arange(len(self.medoid_indices_), dtype=int)
        return self

    def _fit_kmeans_dba(self) -> None:
        """Melatih K-Means time series dengan pembaruan pusat berbasis DBA."""
        best = None
        for run in range(max(1, int(self.n_init))):
            medoids = _init_medoids_pp(self.distance_matrix_, int(self.n_clusters), self.random_state + run)
            centers = self.X_fit_[medoids].copy()
            previous_labels = np.full(len(self.X_fit_), -1, dtype=int)
            for iteration in range(int(self.max_iter)):
                distances = compute_cross_distance(self.X_fit_, centers, self.radius)
                labels = distances.argmin(axis=1)
                updated = centers.copy()
                for cluster_id in range(int(self.n_clusters)):
                    members = np.flatnonzero(labels == cluster_id)
                    if members.size:
                        updated[cluster_id] = _weighted_dba(
                            self.X_fit_[members], np.ones(members.size), centers[cluster_id],
                            self.radius, self.dba_iterations,
                        )
                    else:
                        updated[cluster_id] = self.X_fit_[np.argmax(distances.min(axis=1))]
                shift = float(np.max(np.abs(updated - centers)))
                centers = updated
                self.n_iter_ = iteration + 1
                if np.array_equal(labels, previous_labels) or shift < self.tolerance:
                    break
                previous_labels = labels.copy()
            distances = compute_cross_distance(self.X_fit_, centers, self.radius)
            labels = distances.argmin(axis=1)
            objective = float(np.square(distances.min(axis=1)).sum())
            candidate = (objective, labels, centers.copy(), iteration + 1)
            if best is None or objective < best[0]:
                best = candidate
        self.objective_, self.labels_, self.cluster_centers_, self.n_iter_ = best

    def _fit_kmedoids(self) -> None:
        """Melatih K-Medoids dengan jarak FastDTW dan beberapa inisialisasi."""
        best = None
        for run in range(max(1, int(self.n_init))):
            medoids = _init_medoids_pp(self.distance_matrix_, int(self.n_clusters), self.random_state + run)
            for iteration in range(int(self.max_iter)):
                labels = np.argmin(self.distance_matrix_[:, medoids], axis=1)
                updated = medoids.copy()
                for cluster_id in range(int(self.n_clusters)):
                    members = np.flatnonzero(labels == cluster_id)
                    if members.size:
                        within = self.distance_matrix_[np.ix_(members, members)]
                        updated[cluster_id] = members[np.argmin(within.sum(axis=1))]
                if np.array_equal(updated, medoids):
                    break
                medoids = updated
            labels = np.argmin(self.distance_matrix_[:, medoids], axis=1)
            objective = float(self.distance_matrix_[np.arange(len(labels)), medoids[labels]].sum())
            candidate = (objective, labels, medoids.copy(), iteration + 1)
            if best is None or objective < best[0]:
                best = candidate
        self.objective_, self.labels_, self.medoid_indices_, self.n_iter_ = best

    def _fit_agglomerative(self) -> None:
        """Melatih Agglomerative Clustering dari matriks jarak precomputed."""
        condensed = squareform(self.distance_matrix_, checks=False)
        tree = linkage(condensed, method=self.linkage_method)
        raw_labels = fcluster(tree, t=int(self.n_clusters), criterion="maxclust")
        _, self.labels_ = np.unique(raw_labels, return_inverse=True)
        self.linkage_matrix_ = tree
        self.n_iter_ = len(self.X_fit_) - 1

    def _fit_dbscan(self) -> None:
        """Melatih DBSCAN pada jarak FastDTW dan menyimpan sampel inti untuk prediksi baru."""
        estimator = DBSCAN(eps=float(self.eps), min_samples=int(self.min_samples), metric="precomputed")
        self.labels_ = estimator.fit_predict(self.distance_matrix_)
        self.core_sample_indices_ = estimator.core_sample_indices_.astype(int)
        self.core_labels_ = self.labels_[self.core_sample_indices_]
        self.dbscan_components_ = self.X_fit_[self.core_sample_indices_]
        self.n_iter_ = 1

    def _fit_fuzzy_cmeans(self) -> None:
        """Melatih Fuzzy C-Means berbasis FastDTW dengan pembaruan pusat menggunakan DBA."""
        medoids = _init_medoids_pp(self.distance_matrix_, int(self.n_clusters), self.random_state)
        centers = self.X_fit_[medoids].copy()
        for iteration in range(int(self.max_iter)):
            distances = compute_cross_distance(self.X_fit_, centers, self.radius)
            membership = memberships_from_distances(distances, float(self.fuzziness))
            updated = centers.copy()
            for cluster_id in range(int(self.n_clusters)):
                weights = membership[:, cluster_id] ** float(self.fuzziness)
                updated[cluster_id] = _weighted_dba(
                    self.X_fit_, weights, centers[cluster_id], self.radius, self.dba_iterations
                )
            shift = float(np.max(np.abs(updated - centers)))
            centers = updated
            self.n_iter_ = iteration + 1
            if shift < self.tolerance:
                break
        final_distances = compute_cross_distance(self.X_fit_, centers, self.radius)
        self.membership_ = memberships_from_distances(final_distances, float(self.fuzziness))
        self.labels_ = self.membership_.argmax(axis=1)
        self.cluster_centers_ = centers
        self.objective_ = float(((self.membership_ ** float(self.fuzziness)) * (final_distances ** 2)).sum())
        self.fpc_ = float((self.membership_ ** 2).sum() / len(self.X_fit_))
        self.partition_entropy_ = float(
            -(self.membership_ * np.log(np.maximum(self.membership_, 1e-15))).sum() / len(self.X_fit_)
        )

    def _fit_fuzzy_cmedoids(self) -> None:
        """Melatih Fuzzy C-Medoids dengan optimasi medoid berbobot keanggotaan fuzzy."""
        medoids = _init_medoids_pp(self.distance_matrix_, int(self.n_clusters), self.random_state)
        distance_squared = self.distance_matrix_ ** 2
        for iteration in range(int(self.max_iter)):
            distances = self.distance_matrix_[:, medoids]
            membership = memberships_from_distances(distances, float(self.fuzziness))
            scores = distance_squared.T @ (membership ** float(self.fuzziness))
            proposed = []
            for cluster_id in range(int(self.n_clusters)):
                for candidate in np.argsort(scores[:, cluster_id]):
                    if int(candidate) not in proposed:
                        proposed.append(int(candidate))
                        break
            updated = np.asarray(proposed, dtype=int)
            if len(updated) != int(self.n_clusters) or np.array_equal(updated, medoids):
                break
            medoids = updated
        distances = self.distance_matrix_[:, medoids]
        self.membership_ = memberships_from_distances(distances, float(self.fuzziness))
        self.labels_ = self.membership_.argmax(axis=1)
        self.medoid_indices_ = medoids
        self.objective_ = float(((self.membership_ ** float(self.fuzziness)) * (distances ** 2)).sum())
        self.fpc_ = float((self.membership_ ** 2).sum() / len(self.X_fit_))
        self.partition_entropy_ = float(
            -(self.membership_ * np.log(np.maximum(self.membership_, 1e-15))).sum() / len(self.X_fit_)
        )
        self.n_iter_ = iteration + 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Menempatkan seri baru ke prototype terdekat; DBSCAN dapat mengembalikan noise -1."""
        if not hasattr(self, "labels_"):
            raise RuntimeError("Estimator belum di-fit.")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("X harus berupa matriks dua dimensi tanpa NaN/inf.")
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Jumlah fitur data baru tidak sama dengan data pelatihan.")
        if self.method == "DBSCAN":
            if len(self.core_sample_indices_) == 0:
                return np.full(len(values), -1, dtype=int)
            distances = compute_cross_distance(values, self.dbscan_components_, self.radius)
            nearest = distances.argmin(axis=1)
            nearest_distance = distances[np.arange(len(values)), nearest]
            predicted = self.core_labels_[nearest].copy()
            predicted[nearest_distance > float(self.eps)] = -1
            return predicted

        if self.cluster_centers_ is not None:
            prototypes = self.cluster_centers_
            cluster_ids = np.arange(len(prototypes), dtype=int)
        else:
            prototypes = self.X_fit_[self.medoid_indices_]
            cluster_ids = self.prototype_cluster_labels_
        distances = compute_cross_distance(values, prototypes, self.radius)
        return cluster_ids[distances.argmin(axis=1)]

    def predict_membership(self, X: np.ndarray) -> np.ndarray:
        """Mengestimasi derajat keanggotaan seri baru untuk metode fuzzy."""
        if self.method not in {"FuzzyCMeans", "FuzzyCMedoids"}:
            raise AttributeError("predict_membership hanya tersedia untuk metode fuzzy.")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("X harus berupa matriks dua dimensi tanpa NaN/inf.")
        prototypes = self.cluster_centers_ if self.method == "FuzzyCMeans" else self.X_fit_[self.medoid_indices_]
        distances = compute_cross_distance(values, prototypes, self.radius)
        return memberships_from_distances(distances, float(self.fuzziness))


def build_clustering_pipeline(
    method: str,
    distance_matrix: np.ndarray,
    radius: int = 2,
    random_state: int = 42,
    **model_params,
) -> Pipeline:
    """Membuat sklearn Pipeline: row-wise Min-Max scaling dilanjutkan estimator FastDTW."""
    estimator = FastDTWClusterer(
        method=method, distance_matrix=distance_matrix, radius=radius,
        random_state=random_state, **model_params,
    )
    return Pipeline([
        ("minmax", RowWiseMinMaxScaler(feature_range=(0.0, 1.0))),
        ("cluster", estimator),
    ])
