"""Paths and tunable constants for the ASL Web App."""

from __future__ import annotations

from pathlib import Path

# --- Directories ---
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
STATIC_DIR = APP_DIR / "static"

LEARNED_SIGNS_PATH = DATA_DIR / "learned_signs.pkl"
TEST_SIGNS_PATH = DATA_DIR / "test_signs.pkl"
BLACKLIST_PATH = DATA_DIR / "blacklist.json"
GENERATED_VIDEOS_DIR = STATIC_DIR / "generated_signs"
EVAL_REPORTS_DIR = APP_DIR / "eval_reports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- WLASL dataset (external, read-only) ---
WLASL_ROOT = Path(r"C:\Users\Alina\Desktop\claude\ASL Project")
WLASL_JSON = WLASL_ROOT / "WLASL_v0.3.json"
WLASL_VIDEOS_DIR = WLASL_ROOT / "videos"

# --- Camera recognizer ---
HAND_FEAT_DIM = 63              # 21 landmarks x 3 coords
FEAT_DIM = HAND_FEAT_DIM * 2    # two hands -> 126D
KNN_K = 5
KNN_MAX_COSINE = 0.45
KNN_MIN_EXAMPLES = 3
SMOOTH_N = 3
SMOOTH_MATCH = 2
BLACKLIST_LIMIT = 5

# --- Glove recognizer (5 flex + MPU6050 -> 11D) ---
GLOVE_FLEX_DIM = 5
GLOVE_IMU_DIM = 6
GLOVE_FEAT_DIM = GLOVE_FLEX_DIM + GLOVE_IMU_DIM   # 11
KNN_MAX_COSINE_GLOVE = 0.55
GLOVE_SMOOTH_FRAMES = 5         # moving average over N frames (noise reduction)

# Raw-value ranges used to normalize each channel to [-1, 1]
GLOVE_FLEX_REF_V = 1.85
GLOVE_FLEX_HALF_RANGE_V = 0.5
GLOVE_ACCEL_HALF_RANGE = 2.0 * 9.81
GLOVE_GYRO_HALF_RANGE = 5.0

# --- Camera + glove fusion ---
FUSION_WEIGHT_CAMERA = 0.6
FUSION_WEIGHT_GLOVE = 0.4
TEACH_DEFAULT_USE_GLOVE = True

# --- Word acceptance ---
WORD_HOLD_FRAMES = 12
WORD_HOLD_RATIO = 0.65
WORD_COOLDOWN_SEC = 1.5
HAND_REQUIRED_FRAMES = 4

TEACH_COUNTDOWN_SEC = 2.0
TEACH_CAPTURE_SEC = 3.0

# --- Server ---
HOST = "0.0.0.0"
PORT = 8000
