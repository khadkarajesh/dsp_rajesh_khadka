import os

import numpy as np
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import StandardScaler

ENCODER = "encoder.joblib"


def handle_categorical_data(dataframe: pd.DataFrame, model_dir: Path):
    columns = get_categorical_columns(dataframe)
    categorical_dataframe = dataframe[columns].copy()
    filled_unknowns_dataframe = fill_unknown(columns, categorical_dataframe)
    encoder = get_encoder(model_dir=model_dir)
    encoder.fit(filled_unknowns_dataframe)
    encoded_sparse_matrix = encoder.transform(filled_unknowns_dataframe)
    encoded_data_columns = encoder.get_feature_names(columns)
    encoded_categorical_data_df = pd.DataFrame.sparse.from_spmatrix(data=encoded_sparse_matrix,
                                                                    columns=encoded_data_columns,
                                                                    index=filled_unknowns_dataframe.index)
    return encoded_categorical_data_df


def get_categorical_columns(dataframe: pd.DataFrame):
    return dataframe.select_dtypes(exclude=[np.number]).columns.tolist()


def get_continuous_columns(dataframe: pd.DataFrame):
    return dataframe.select_dtypes(exclude=[np.object]).columns.tolist()


def handle_numerical_data(dataframe: pd.DataFrame):
    columns = get_continuous_columns(dataframe)
    imputed_dataframe = dataframe[columns].copy()

    simple_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputed_dataframe[columns] = simple_imputer.fit_transform(imputed_dataframe[columns])

    scalar = StandardScaler()
    imputed_dataframe[columns] = scalar.fit_transform(imputed_dataframe[columns])

    return imputed_dataframe


def preprocess(X: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    numerical_dataframe = handle_numerical_data(dataframe=X)
    categorical_dataframe = handle_categorical_data(dataframe=X, model_dir=model_dir)
    merged_dataframe = numerical_dataframe.join(categorical_dataframe)
    return merged_dataframe


def get_encoder(model_dir: Path) -> OneHotEncoder:
    encoder_path = Path(model_dir / ENCODER)
    if encoder_path.exists(): return joblib.load(encoder_path)
    encoder = OneHotEncoder(handle_unknown='ignore', dtype=int, sparse=True)
    joblib.dump(encoder, model_dir / ENCODER)
    return encoder


def get_column_transformer(path_to_save_encoder) -> ColumnTransformer:
    numerical_preprocessor = Pipeline(steps=[('imputer', SimpleImputer()),
                                             ('scaler', MinMaxScaler())])
    categorical_preprocessor = Pipeline(
        steps=[('encoder', get_encoder(path_to_save_encoder))])
    column_transformer = ColumnTransformer(
        transformers=[('num', numerical_preprocessor, selector(dtype_exclude="object")),
                      ('cat', categorical_preprocessor, selector(dtype_exclude="int64"))],
        remainder='passthrough')
    return column_transformer


def fill_unknown(columns: List, dataframe) -> pd.DataFrame:
    for column in columns:
        dataframe[column] = dataframe[column].fillna('Unknown')
    return dataframe


def fill_with_most_frequent(columns: List, dataframe) -> pd.DataFrame:
    for column in columns:
        dataframe[column].fillna(dataframe.value_counts().index[0])
    return dataframe


def impute_numerical_value(columns: List, dataframe, **kwargs) -> pd.DataFrame:
    value = 0
    for column in columns:
        if kwargs.get('type') == 'mean':
            value = dataframe[column].mean()
        elif kwargs.get('type') == 'median':
            value = dataframe[column].median()
        dataframe[column].fillna(value)
    return dataframe
