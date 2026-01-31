"""
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
