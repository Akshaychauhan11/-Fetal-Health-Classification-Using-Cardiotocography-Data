import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from src.data.load import load_data, TARGET_COLUMN


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Perform a two-step stratified train/calibration/test split.

    Split ratio: 60% Train / 20% Calibration / 20% Test

    Step 1: Split entire dataset into 60% Train and 40% Temp (Calibration + Test).
    Step 2: Split 40% Temp 50/50 into 20% Calibration and 20% Test.

    Parameters
    ----------
    df : pd.DataFrame
        Complete CTG dataset.
    target_col : str
        Name of the target column ('fetal_health').
    random_state : int
        Seed for reproducibility (default: 42).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
        X_train, X_cal, X_test, y_train, y_cal, y_test
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Step 1: 60% Train, 40% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.40,
        stratify=y,
        random_state=random_state
    )

    # Step 2: 50% Calibration, 50% Test (from the 40% Temp set -> 20% each of total)
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state
    )

    print(f"[src.data.split] Stratified Split Completed (random_state={random_state}):")
    print(f"  Train:       {X_train.shape[0]} samples ({X_train.shape[0]/len(df):.1%})")
    print(f"  Calibration: {X_cal.shape[0]} samples ({X_cal.shape[0]/len(df):.1%})")
    print(f"  Test:        {X_test.shape[0]} samples ({X_test.shape[0]/len(df):.1%})")

    return X_train, X_cal, X_test, y_train, y_cal, y_test


if __name__ == "__main__":
    data = load_data()
    X_tr, X_c, X_te, y_tr, y_c, y_te = split_data(data)

    print("\nClass distribution across splits (percentages):")
    split_df = pd.DataFrame({
        'Train %': y_tr.value_counts(normalize=True) * 100,
        'Calibration %': y_c.value_counts(normalize=True) * 100,
        'Test %': y_te.value_counts(normalize=True) * 100
    })
    print(split_df.round(2))
