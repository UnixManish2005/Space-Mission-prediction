"""
utils.py - Utility Functions for Space Mission Success Prediction
=================================================================
Contains helper functions for data loading, preprocessing,
feature engineering, and model evaluation.
"""

import pandas as pd
import numpy as np
import warnings
import re
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    'rocket_cost', 'launch_year', 'launch_month', 'rocket_status_encoded',
    'company_encoded', 'location_encoded', 'country_encoded', 'decade',
    'company_success_rate', 'company_launch_count', 'rocket_encoded'
]

COMPANY_LIST = []
LOCATION_LIST = []
COUNTRY_LIST = []
ROCKET_LIST = []


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the Space_Corrected.csv dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Raw DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        # Strip leading/trailing spaces from column names
        df.columns = df.columns.str.strip()
        print(f"[✓] Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")


# ──────────────────────────────────────────────
# DATA CLEANING
# ──────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleaning steps:
      - Remove duplicates
      - Drop irrelevant columns
      - Rename columns for consistency
      - Parse rocket cost to numeric
      - Parse dates

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    # ---- Remove duplicates ----
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[✓] Removed {before - len(df)} duplicate rows")

    # ---- Standardise column names ----
    rename_map = {}
    for col in df.columns:
        stripped = col.strip()
        rename_map[col] = stripped
    df.rename(columns=rename_map, inplace=True)

    # ---- Drop useless index column ----
    for drop_col in ['Unnamed: 0', 'Unnamed: 0.1']:
        if drop_col in df.columns:
            df.drop(columns=[drop_col], inplace=True)

    # ---- Clean rocket cost ----
    if 'Rocket' in df.columns:
        df['rocket_cost'] = (
            df['Rocket']
            .astype(str)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        df['rocket_cost'] = pd.to_numeric(df['rocket_cost'], errors='coerce')
    else:
        df['rocket_cost'] = np.nan

    # ---- Parse date / extract year & month ----
    if 'Datum' in df.columns:
        df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
        df['launch_year']  = df['Datum'].dt.year
        df['launch_month'] = df['Datum'].dt.month
    else:
        df['launch_year']  = np.nan
        df['launch_month'] = np.nan

    print(f"[✓] Cleaning done. Shape: {df.shape}")
    return df


# ──────────────────────────────────────────────
# TARGET VARIABLE
# ──────────────────────────────────────────────

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target variable from 'Status Mission':
      'Success' → 1, everything else → 0

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with a new 'target' column.
    """
    df = df.copy()
    col = 'Status Mission'
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset.")
    df['target'] = df[col].apply(lambda x: 1 if str(x).strip() == 'Success' else 0)
    print(f"[✓] Target distribution:\n{df['target'].value_counts()}")
    return df


# ──────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────

def extract_country(location: str) -> str:
    """Extract the country from a location string (last part after last comma)."""
    if pd.isna(location) or location == '':
        return 'Unknown'
    parts = [p.strip() for p in str(location).split(',')]
    return parts[-1] if parts else 'Unknown'


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features:
      - Country from Location
      - Decade
      - Company success rate (target-encoded mean)
      - Company launch count
      - Label-encode categorical cols

    Args:
        df: Cleaned DataFrame with target column.

    Returns:
        Feature-engineered DataFrame.
    """
    global COMPANY_LIST, LOCATION_LIST, COUNTRY_LIST, ROCKET_LIST

    df = df.copy()

    # ---- Country ----
    loc_col = 'Location' if 'Location' in df.columns else None
    if loc_col:
        df['country'] = df[loc_col].apply(extract_country)
    else:
        df['country'] = 'Unknown'

    # ---- Decade ----
    df['decade'] = (df['launch_year'] // 10 * 10).fillna(0).astype(int)

    # ---- Rocket status encode ----
    rs_col = 'Status Rocket' if 'Status Rocket' in df.columns else None
    if rs_col:
        df['rocket_status_encoded'] = df[rs_col].apply(
            lambda x: 1 if str(x).strip() == 'StatusActive' else 0
        )
    else:
        df['rocket_status_encoded'] = 0

    # ---- Company success rate (leave-one-out mean target) ----
    cn_col = 'Company Name' if 'Company Name' in df.columns else None
    if cn_col:
        company_stats = df.groupby(cn_col)['target'].agg(['mean', 'count']).reset_index()
        company_stats.columns = [cn_col, 'company_success_rate', 'company_launch_count']
        df = df.merge(company_stats, on=cn_col, how='left')
        COMPANY_LIST = sorted(df[cn_col].dropna().unique().tolist())
    else:
        df['company_success_rate'] = 0.5
        df['company_launch_count'] = 0

    # ---- Label encoding ----
    def label_encode(series, global_list_ref):
        cats = sorted(series.dropna().unique().tolist())
        global_list_ref.clear()
        global_list_ref.extend(cats)
        mapping = {cat: i for i, cat in enumerate(cats)}
        return series.map(mapping).fillna(-1).astype(int)

    if cn_col:
        df['company_encoded'] = label_encode(df[cn_col], COMPANY_LIST)
    else:
        df['company_encoded'] = -1

    if loc_col:
        df['location_encoded'] = label_encode(df[loc_col], LOCATION_LIST)
    else:
        df['location_encoded'] = -1

    df['country_encoded'] = label_encode(df['country'], COUNTRY_LIST)

    detail_col = 'Detail' if 'Detail' in df.columns else None
    if detail_col:
        df['rocket_encoded'] = label_encode(df[detail_col], ROCKET_LIST)
    else:
        df['rocket_encoded'] = -1

    # ---- Fill missing numerics ----
    df['rocket_cost']   = df['rocket_cost'].fillna(df['rocket_cost'].median())
    df['launch_year']   = df['launch_year'].fillna(df['launch_year'].median())
    df['launch_month']  = df['launch_month'].fillna(6)

    print(f"[✓] Feature engineering done. Features: {FEATURE_COLUMNS}")
    return df


def get_features_target(df: pd.DataFrame):
    """
    Extract X (features) and y (target) arrays.

    Returns:
        X (DataFrame), y (Series)
    """
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available].copy()
    y = df['target'].copy()
    return X, y


# ──────────────────────────────────────────────
# PREPROCESSING PIPELINE (used in app.py)
# ──────────────────────────────────────────────

def preprocess_pipeline(filepath: str):
    """
    Full preprocessing pipeline that returns ready-to-train X, y and the
    processed DataFrame.

    Args:
        filepath: Path to raw CSV.

    Returns:
        X, y, processed_df
    """
    df = load_data(filepath)
    df = clean_data(df)
    df = create_target(df)
    df = feature_engineering(df)
    X, y = get_features_target(df)
    return X, y, df


# ──────────────────────────────────────────────
# METRICS HELPER
# ──────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Compute classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (optional, for ROC-AUC).

    Returns:
        Dictionary of metrics.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )

    metrics = {
        'accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y_true, y_pred, zero_division=0), 4),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None:
        try:
            metrics['roc_auc'] = round(roc_auc_score(y_true, y_prob), 4)
        except Exception:
            metrics['roc_auc'] = None
    return metrics


# ──────────────────────────────────────────────
# SINGLE PREDICTION HELPER
# ──────────────────────────────────────────────

def build_input_row(
    company: str,
    location: str,
    rocket_name: str,
    rocket_status: str,
    rocket_cost: float,
    launch_year: int,
    launch_month: int,
    encodings: dict,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame for model prediction from user inputs.

    Args:
        company: Company name.
        location: Launch location string.
        rocket_name: Rocket / mission detail.
        rocket_status: 'StatusActive' or 'StatusRetired'.
        rocket_cost: Cost in millions USD.
        launch_year: Year of launch.
        launch_month: Month of launch.
        encodings: Dict of label-encoding maps loaded from trained artefact.

    Returns:
        Single-row DataFrame matching FEATURE_COLUMNS.
    """
    country = extract_country(location)
    decade  = int(launch_year // 10 * 10)

    # Encode categoricals (fallback = -1 if unseen)
    company_enc  = encodings.get('company_map',  {}).get(company,  -1)
    location_enc = encodings.get('location_map', {}).get(location, -1)
    country_enc  = encodings.get('country_map',  {}).get(country,  -1)
    rocket_enc   = encodings.get('rocket_map',   {}).get(rocket_name, -1)
    rs_enc       = 1 if rocket_status == 'StatusActive' else 0

    # Company stats
    csr = encodings.get('company_success_rate', {}).get(company, 0.5)
    clc = encodings.get('company_launch_count', {}).get(company, 0)

    row = {
        'rocket_cost':           rocket_cost,
        'launch_year':           launch_year,
        'launch_month':          launch_month,
        'rocket_status_encoded': rs_enc,
        'company_encoded':       company_enc,
        'location_encoded':      location_enc,
        'country_encoded':       country_enc,
        'decade':                decade,
        'company_success_rate':  csr,
        'company_launch_count':  clc,
        'rocket_encoded':        rocket_enc,
    }
    return pd.DataFrame([row])
