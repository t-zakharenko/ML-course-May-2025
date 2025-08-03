import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Tuple, List, Optional


def split_dataset(df: pd.DataFrame, target_col: str, test_size: float = 0.25, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and validation sets using stratification.

    Args:
        df (pd.DataFrame): Full raw dataset.
        target_col (str): Name of the target column for stratification.
        test_size (float): Validation set size.
        random_state (int): Random seed.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and validation sets.
    """
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])


def encode_gender_column(df: pd.DataFrame, col: str = 'Gender') -> pd.DataFrame:
    """
    Convert the gender column to binary: 1 for 'Male', 0 otherwise.

    Args:
        df (pd.DataFrame): DataFrame containing gender column.
        col (str): Name of the gender column.

    Returns:
        pd.DataFrame: DataFrame with binary gender column.
    """
    df[col] = (df[col] == 'Male').astype(int)
    return df


def scale_numeric_features(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Apply MinMax scaling to numeric features (excluding binary).

    Args:
        train_df (pd.DataFrame): Training features.
        val_df (pd.DataFrame): Validation features.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]: Scaled train/val sets and fitted scaler.
    """
    numeric_cols = train_df.select_dtypes(include='number').columns.tolist()
    binary_cols = [col for col in numeric_cols if train_df[col].nunique() == 2]
    scale_cols = [col for col in numeric_cols if col not in binary_cols]

    scaler = MinMaxScaler().fit(train_df[scale_cols])
    train_df[scale_cols] = scaler.transform(train_df[scale_cols])
    val_df[scale_cols] = scaler.transform(val_df[scale_cols])

    return train_df, val_df, scaler


def preprocess_data(
    raw_df: pd.DataFrame,
    target_col: str = 'Exited',
    scaler_numeric: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], Optional[MinMaxScaler], OneHotEncoder]:
    """
    Full preprocessing pipeline: splits data, encodes, optionally scales.

    Args:
        raw_df (pd.DataFrame): Raw dataset.
        target_col (str): Target column name.
        scaler_numeric (bool): Whether to scale numeric features.

    Returns:
        Tuple containing:
            X_train (pd.DataFrame)
            y_train (pd.Series)
            X_val (pd.DataFrame)
            y_val (pd.Series)
            input_cols (List[str])
            scaler (Optional[MinMaxScaler])
            encoder (OneHotEncoder)
    """
    # Drop Surname if exists
    raw_df = raw_df.drop(columns='Surname', errors='ignore')

    # Train/Val split
    train_df, val_df = split_dataset(raw_df, target_col)
    input_cols = list(train_df.columns)[3:-1]

    X_train = train_df[input_cols].copy()
    y_train = train_df[target_col].copy()
    X_val = val_df[input_cols].copy()
    y_val = val_df[target_col].copy()

    # One-hot encode 'Geography'
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train[['Geography']])

    geo_col_names = list(encoder.get_feature_names_out(['Geography']))
    X_train[geo_col_names] = encoder.transform(X_train[['Geography']])
    X_val[geo_col_names] = encoder.transform(X_val[['Geography']])
    X_train.drop(columns='Geography', inplace=True)
    X_val.drop(columns='Geography', inplace=True)

    # Encode Gender
    X_train = encode_gender_column(X_train)
    X_val = encode_gender_column(X_val)

    # Scale if enabled
    scaler = None
    if scaler_numeric:
        X_train, X_val, scaler = scale_numeric_features(X_train, X_val)

    return X_train, y_train, X_val, y_val, X_train.columns.tolist(), scaler, encoder


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    encoder: OneHotEncoder,
    scaler: Optional[MinMaxScaler] = None
) -> pd.DataFrame:
    """
    Preprocess new data using pre-fitted encoder and optional scaler.

    Args:
        new_df (pd.DataFrame): Raw new data.
        input_cols (List[str]): Columns to use.
        encoder (OneHotEncoder): Pre-fitted encoder.
        scaler (MinMaxScaler, optional): Pre-fitted scaler.

    Returns:
        pd.DataFrame: Preprocessed data ready for prediction.
    """
    df = new_df.copy()

    # Drop Surname if exists
    df = df.drop(columns='Surname', errors='ignore')

    # Select features
    df = df[input_cols].copy()

    # One-hot encode 'Geography'
    geo_col_names = list(encoder.get_feature_names_out(['Geography']))
    df[geo_col_names] = encoder.transform(df[['Geography']])
    df.drop(columns='Geography', inplace=True)

    # Encode Gender
    df = encode_gender_column(df)

    # Scale if scaler is provided
    if scaler:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        binary_cols = [col for col in numeric_cols if df[col].nunique() == 2]
        scale_cols = [col for col in numeric_cols if col not in binary_cols]
        df[scale_cols] = scaler.transform(df[scale_cols])

    return df
