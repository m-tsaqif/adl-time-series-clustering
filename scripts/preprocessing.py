"""Preprocessing berbasis data penjualan asli dengan Min-Max per produk."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class RowWiseMinMaxScaler(BaseEstimator, TransformerMixin):
    """Min-Max scaling per baris/produk agar jarak berfokus pada bentuk pola mingguan."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0), constant_value: float = 0.0):
        """Menginisialisasi rentang target dan nilai konstan untuk seri datar."""
        self.feature_range = feature_range
        self.constant_value = constant_value

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "RowWiseMinMaxScaler":
        """Memvalidasi jumlah fitur dan rentang target untuk transformasi per produk."""
        array = np.asarray(X, dtype=float)
        if array.ndim != 2 or array.size == 0 or not np.isfinite(array).all():
            raise ValueError("X harus berupa matriks dua dimensi tanpa NaN/inf.")
        low, high = self.feature_range
        if not low < high:
            raise ValueError("feature_range harus memiliki batas bawah lebih kecil dari batas atas.")
        self.n_features_in_ = array.shape[1]
        self.feature_names_in_ = None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Menskalakan setiap seri mingguan ke rentang target secara independen."""
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("Scaler belum di-fit.")
        array = np.asarray(X, dtype=float)
        if array.ndim != 2 or array.size == 0 or not np.isfinite(array).all():
            raise ValueError("X harus berupa matriks dua dimensi tanpa NaN/inf.")
        if array.shape[1] != self.n_features_in_:
            raise ValueError(f"Jumlah minggu berbeda: diharapkan {self.n_features_in_}, diterima {array.shape[1]}.")
        low, high = self.feature_range
        row_min = array.min(axis=1, keepdims=True)
        row_range = array.max(axis=1, keepdims=True) - row_min
        safe_range = np.where(row_range > np.finfo(float).eps, row_range, 1.0)
        scaled = (array - row_min) / safe_range
        scaled = low + scaled * (high - low)
        constant_rows = row_range[:, 0] <= np.finfo(float).eps
        if constant_rows.any():
            scaled[constant_rows, :] = self.constant_value
        return np.asarray(scaled, dtype=np.float64)


def build_preprocessing_pipeline(feature_range: tuple[float, float] = (0.0, 1.0)) -> Pipeline:
    """Membuat pipeline preprocessing agar transformasi dapat di-fit dan dipakai ulang."""
    return Pipeline([("row_minmax", RowWiseMinMaxScaler(feature_range=feature_range))])


def transform_sales_data(
    values: np.ndarray,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[Pipeline, np.ndarray]:
    """Fit pipeline Min-Max per produk pada W0–W51 asli dan mengembalikan fitur transformasi."""
    pipeline = build_preprocessing_pipeline(feature_range=feature_range)
    transformed = pipeline.fit_transform(values)
    return pipeline, transformed
