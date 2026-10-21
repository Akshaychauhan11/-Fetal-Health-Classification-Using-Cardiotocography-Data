import os
import pandas as pd
from typing import Union
from pathlib import Path


EXPECTED_COLUMNS = [
    'baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions',
    'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
    'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability',
    'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min',
    'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes',
    'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance',
    'histogram_tendency', 'fetal_health'
]

TARGET_COLUMN = 'fetal_health'


def load_data(data_path: Union[str, Path] = "data/raw/fetal_health.csv") -> pd.DataFrame:
    """
    Load and lightly validate the Cardiotocography (CTG) fetal health dataset.

    Parameters
    ----------
    data_path : str or Path
        Path to the raw CSV file. Defaults to 'data/raw/fetal_health.csv'.

    Returns
    -------
    pd.DataFrame
        Loaded and validated CTG dataset.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist at data_path.
    ValueError
        If expected target column is missing, unexpected columns exist, or null values are present.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")

    df = pd.read_csv(path)

    # Validate shape and columns
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataset.")

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing expected columns: {missing_cols}")

    # Validate missing / null values
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        raise ValueError(f"Dataset contains {null_count} missing (null) values.")

    print(f"[src.data.load] Dataset loaded successfully from '{path}'. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    # Quick test execution
    data = load_data()
    print(f"Target distribution:\n{data[TARGET_COLUMN].value_counts()}")
