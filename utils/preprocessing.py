import pandas as pd

def load_data(path: str):
    """Load dataset from a CSV file."""
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame):
    """Basic cleaning (drop NA values)."""
    return df.dropna()
