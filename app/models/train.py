import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

from app.constant import AppConstant
from app.models.inference import predict
from app.models.preprocess import preprocess


def save_model(model, model_dir):
    file_name = model_dir / AppConstant.MODEL_NAME
    joblib.dump(model, file_name)
    return file_name.name


def train_model(training_data_filepath: str, model_dir):
    x_train, x_test, y_test, y_train = spilit_data(model_dir, training_data_filepath)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    model_path = save_model(model, model_dir)

    y_predicted = predict(x_test, model_dir)

    return {"model_performance": mean_squared_log_error(y_test, y_predicted),
            'path_to_model': str(model_dir / model_path)}


def spilit_data(model_dir, training_data_filepath):
    dataframe = pd.read_csv(Path(training_data_filepath))
    df = dataframe.copy()
    y = df['SalePrice']
    x = preprocess(df, model_dir)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_test, y_train
