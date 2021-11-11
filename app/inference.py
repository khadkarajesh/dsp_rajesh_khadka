from pathlib import Path

import joblib
import numpy as np

from app.constant import AppConstant
from app.preprocess import preprocess


def predict(dataframe: str = None, model_dir: Path = None) -> np.ndarray:
    processed_data = preprocess(dataframe, model_dir)
    model = joblib.load(model_dir / AppConstant.MODEL_NAME)
    return model.predict(processed_data)
