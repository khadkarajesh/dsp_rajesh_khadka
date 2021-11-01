import os
from pathlib import Path

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error

from app.models.preprocess import fill_unknown, get_column_transformer


def save_model(model):
    path = Path("../../app/models/")
    file_name = path / 'model.joblib'
    joblib.dump(model, file_name)
    return file_name.name


def train_model(dataset_path, column_transformer):
    dataframe = pd.read_csv(dataset_path)
    df = dataframe.copy()

    categorical_columns = df.select_dtypes(include='object').columns
    df = fill_unknown(categorical_columns, df)

    df.drop('SalePrice', axis=1)
    X = df
    y = df['SalePrice']

    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    pipeline = Pipeline(
        steps=[("preprocessor", column_transformer), ("model", LogisticRegression())]
    )
    pipeline.fit(X_train, y_train)
    y_predicted = pipeline.predict(x_test)
    mean_squared_log_error(y_test, y_predicted)

    return {"model_performance": mean_squared_log_error(y_test, y_predicted), 'path_to_model': save_model(pipeline)}


print(train_model(Path("../../data/house-prices", "train.csv"), get_column_transformer()))
