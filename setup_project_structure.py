"""
Run this script to create the complete src directory structure
Save as: setup_project_structure.py
Run from project root: python setup_project_structure.py
"""

import os
from pathlib import Path

def create_project_structure():
    """Create complete project directory structure"""
    
    # Find project root
    project_root = Path.cwd()
    while project_root.name != 'bpo-intelligence-platform' and project_root.parent != project_root:
        project_root = project_root.parent
    
    if project_root.name != 'bpo-intelligence-platform':
        project_root = Path.cwd()
    
    print(f"Setting up project structure in: {project_root}\n")
    
    # Define directory structure
    directories = [
        'src',
        'src/data_pipeline',
        'src/models',
        'src/models/ml',
        'src/models/dl',
        'src/models/nlp',
        'src/models/genai',
        'src/models/agentic',
        'src/utils',
        'src/api',
        'notebooks',
        'notebooks/01_eda',
        'notebooks/02_ml',
        'notebooks/03_dl',
        'notebooks/04_nlp',
        'notebooks/05_genai',
        'notebooks/06_agentic',
        'data/raw',
        'data/processed',
        'data/features',
        'data/external',
        'models/saved_models',
        'models/checkpoints',
        'reports/figures',
        'reports/metrics',
        'tests',
        'config',
        'deployment/aws',
        'deployment/docker',
        'docs'
    ]
    
    # Create directories
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    print("\n" + "="*60)
    print("Creating __init__.py files...")
    print("="*60 + "\n")
    
    # Create __init__.py files
    init_files = [
        'src/__init__.py',
        'src/data_pipeline/__init__.py',
        'src/models/__init__.py',
        'src/models/ml/__init__.py',
        'src/models/dl/__init__.py',
        'src/models/nlp/__init__.py',
        'src/models/genai/__init__.py',
        'src/models/agentic/__init__.py',
        'src/utils/__init__.py',
        'src/api/__init__.py',
        'tests/__init__.py'
    ]
    
    for init_file in init_files:
        init_path = project_root / init_file
        if not init_path.exists():
            init_path.write_text('"""Package initialization"""\n', encoding='utf-8')
            print(f"✅ Created: {init_file}")
    
    print("\n" + "="*60)
    print("Creating essential Python files...")
    print("="*60 + "\n")
    
    # Create config.py
    config_content = '''"""
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
'''
    
    config_path = project_root / 'src' / 'config.py'
    config_path.write_text(config_content, encoding='utf-8')
    print(f"✅ Created: src/config.py")
    
    # Create utils.py
    utils_content = '''"""
Utility functions for BPO Intelligence Platform
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json


def load_data(data_path):
    """Load CSV data"""
    return pd.read_csv(data_path)


def save_model(model, filepath):
    """Save model using joblib"""
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath):
    """Load model using joblib"""
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model


def plot_feature_importance(feature_importance_df, top_n=20, figsize=(10, 8)):
    """Plot feature importance"""
    top_features = feature_importance_df.nlargest(top_n, 'importance')
    
    plt.figure(figsize=figsize)
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title(f'Top {top_n} Important Features')
    plt.xlabel('Importance')
    plt.tight_layout()
    return plt.gcf()


def save_metrics(metrics_dict, filepath):
    """Save metrics to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics saved to: {filepath}")


def load_metrics(filepath):
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    return metrics


def create_submission_df(predictions, ids, target_col='prediction'):
    """Create submission dataframe"""
    return pd.DataFrame({
        'id': ids,
        target_col: predictions
    })


print("Utils module loaded")
'''
    
    utils_path = project_root / 'src' / 'utils' / 'helpers.py'
    utils_path.write_text(utils_content, encoding='utf-8')
    print(f"✅ Created: src/utils/helpers.py")
    
    # Create .gitkeep files in empty directories
    gitkeep_dirs = [
        'data/raw',
        'data/processed',
        'data/features',
        'data/external',
        'models/saved_models',
        'models/checkpoints',
        'reports/figures',
        'reports/metrics'
    ]
    
    print("\n" + "="*60)
    print("Creating .gitkeep files...")
    print("="*60 + "\n")
    
    for directory in gitkeep_dirs:
        gitkeep_path = project_root / directory / '.gitkeep'
        gitkeep_path.write_text('', encoding='utf-8')
        print(f"✅ Created: {directory}/.gitkeep")
    
    print("\n" + "="*60)
    print("✅ PROJECT STRUCTURE SETUP COMPLETE!")
    print("="*60)
    print(f"\nProject root: {project_root}")
    print("\nYou can now:")
    print("1. Import modules: from src.data_pipeline.feature_engineering import BPOFeatureEngineer")
    print("2. Use config: from src.config import DATA_RAW, MODELS_SAVED")
    print("3. Use utils: from src.utils.helpers import save_model, load_data")
    print("\n" + "="*60)


if __name__ == "__main__":
    create_project_structure()