import joblib
import numpy as np
import pandas as pd


def predict(data_frame: pd.DataFrame, data_filepath: str = None) -> np.ndarray:
    pipeline = joblib.load(data_filepath)
    return pipeline.predict(data_frame)
