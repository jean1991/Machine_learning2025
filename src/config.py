"""
Configuration settings for the Smart Governance ML Solution.

This module contains all configuration parameters, file paths, and settings
for the machine learning pipeline and dashboard.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_PATH = DATA_DIR / "sample_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "resource_allocation_model.joblib"

# Notebook paths
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
NOTEBOOKS_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "model_type": "random_forest",
    "n_estimators": 100,
    "max_depth": 10
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    "title": "Smart Governance Analytics Dashboard",
    "page_icon": "🏛️",
    "layout": "wide"
}

# Data preprocessing configuration
PREPROCESSING_CONFIG = {
    "categorical_columns": ["region", "district"],
    "numerical_columns": ["population", "income_per_capita", "health_index", "education_index"],
    "target_column": "current_allocation"
}

# Supported regions and districts for validation
VALID_REGIONS = ["East", "North", "West", "Central"]

VALID_DISTRICTS = [
    "Kampala", "Jinja", "Mbale", "Gulu", "Lira", "Mbarara", "Kasese", 
    "Masaka", "Mukono", "Soroti", "Arua", "Hoima", "Tororo", "Kitgum",
    "Wakiso", "Kabale", "Iganga", "Moroto", "Bundibugyo", "Mpigi",
    "Busia", "Adjumani", "Kibaale", "Mityana", "Pallisa"
]