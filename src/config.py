"""
Project Configuration
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_FEATURES = DATA_DIR / "features"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_SAVED = MODELS_DIR / "saved_models"
MODELS_CHECKPOINTS = MODELS_DIR / "checkpoints"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_FIGURES = REPORTS_DIR / "figures"
REPORTS_METRICS = REPORTS_DIR / "metrics"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature engineering
ROLLING_WINDOWS = [7, 14, 30]
LAG_FEATURES = [1, 7, 14]

print(f"Configuration loaded from: {PROJECT_ROOT}")
