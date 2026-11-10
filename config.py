"""Konfigurasi terpusat untuk analisis clustering pola penjualan."""
from pathlib import Path
import os

# Lokasi proyek dan data
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "dataset"
NOTEBOOK_DIR = PROJECT_ROOT / "notebook"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODEL_DIR = PROJECT_ROOT / "model"
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOT_DIR = OUTPUT_DIR / "plots"
DISTANCE_CACHE_DIR = OUTPUT_DIR / "distance_cache"
DATA_FILE = DATA_DIR / "sales.csv"

# Struktur data
ID_COLUMN = "Product_Code"
WEEK_COLUMNS = [f"W{i}" for i in range(52)]
EXPECTED_N_WEEKS = len(WEEK_COLUMNS)

# Reproducibility dan preprocessing
RANDOM_STATE = 42
MINMAX_FEATURE_RANGE = (0.0, 1.0)
MINMAX_CONSTANT_VALUE = 0.0
# Row-wise: setiap produk dinormalisasi secara independen untuk membandingkan bentuk pola.
MINMAX_AXIS = "per_product"

# FastDTW
FASTDTW_RADIUS = 2
FASTDTW_LOCAL_COST = "squared_difference"
NUMBA_THREADS = max(1, min(4, os.cpu_count() or 1))
REUSE_DISTANCE_CACHE = True

# Tuning enam metode
K_GRID = [2]
LINKAGE_GRID = ["average", "complete"]
DBSCAN_MIN_SAMPLES_GRID = [4, 7, 15, 25]
DBSCAN_EPS_QUANTILES = [0.10, 0.20, 0.30, 0.40, 0.55, 0.75, 0.93]
FUZZY_M_GRID = [1.7, 2.0]

# Iterasi / inisialisasi
KMEANS_MAX_ITER = 10
KMEANS_N_INIT = 1
DBA_ITERATIONS = 2
KMEDOIDS_MAX_ITER = 25
KMEDOIDS_N_INIT = 1
FCM_MAX_ITER = 4
FCM_DBA_ITERATIONS = 1
FCM_TOLERANCE = 1e-4
FCMEDOIDS_MAX_ITER = 25
STABILITY_SEEDS = [42, 52, 62]
STABILITY_ARI_MIN = 0.75
SILHOUETTE_TUNING_SAMPLE_SIZE = 350
VERBOSE_EXPERIMENT = True

# Evaluasi dan pemilihan model
MIN_CLUSTER_SHARE = 0.01
MIN_COVERAGE_PRIMARY = 1.0
MAX_NOISE_SHARE_FOR_TUNING = 0.40
SHAPE_MAX_ETA_ZERO = 0.10
DEFAULT_METHOD = "KMedoids"
# Bobot ranking komposit untuk membantu menyaring kandidat, bukan pengganti penilaian substantif.
METRIC_WEIGHTS = {
    "Silhouette_FastDTW": (0.40, True),
    "DBI_Medoid_FastDTW": (0.30, False),
    "CHI_MinMax_Euclidean": (0.15, True),
    "DBI_MinMax_Euclidean": (0.10, False),
    "Coverage": (0.05, True),
}

# Tampilan
PLOT_TEMPLATE = "none"
PLOT_FONT = "Plus Jakarta Sans, Arial"
PLOT_PRIMARY = "#1F6B4F"
PLOT_ACCENT = "#C39231"
PLOT_PALETTE = ["#1F6B4F", "#C39231", "#6285A0", "#9A6775", "#8A78A1"]
PLOT_BACKGROUND = "rgba(0,0,0,0)"

# Ekspor
SAVE_PLOT_HTML = True
SAVE_PLOT_JSON = False
JOBLIB_COMPRESSION = 3
