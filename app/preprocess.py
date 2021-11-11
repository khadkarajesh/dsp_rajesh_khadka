from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from app.constant import AppConstant


def __handle_categorical_data(dataframe: pd.DataFrame, model_dir: Path):
    columns = __get_categorical_columns(dataframe)
    categorical_dataframe = dataframe[columns].copy()
    filled_unknowns_dataframe = __fill_unknown(columns, categorical_dataframe)
    encoder = __get_encoder(model_dir=model_dir)
    encoder.fit(filled_unknowns_dataframe)
    encoded_sparse_matrix = encoder.transform(filled_unknowns_dataframe)
    encoded_data_columns = encoder.get_feature_names(columns)
    encoded_categorical_data_df = pd.DataFrame.sparse.from_spmatrix(data=encoded_sparse_matrix,
                                                                    columns=encoded_data_columns,
                                                                    index=filled_unknowns_dataframe.index)
    return encoded_categorical_data_df


def __get_categorical_columns(dataframe: pd.DataFrame):
    return dataframe.select_dtypes(exclude=[np.number]).columns.tolist()


def __get_continuous_columns(dataframe: pd.DataFrame):
    return dataframe.select_dtypes(exclude=[np.object]).columns.tolist()


def __handle_numerical_data(dataframe: pd.DataFrame):
    columns = __get_continuous_columns(dataframe)
    imputed_dataframe = dataframe[columns].copy()

    imputed_dataframe = __impute_numerical_value(columns, imputed_dataframe)
    scaled_dataframe = __scale_numerical_data(columns, imputed_dataframe)

    return scaled_dataframe


def __scale_numerical_data(columns, imputed_dataframe):
    scalar = StandardScaler()
    imputed_dataframe[columns] = scalar.fit_transform(imputed_dataframe[columns])
    return imputed_dataframe


def __impute_numerical_value(columns, imputed_dataframe):
    simple_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputed_dataframe[columns] = simple_imputer.fit_transform(imputed_dataframe[columns])
    return imputed_dataframe


def preprocess(X: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    columns_to_drop = ['Id', 'SalePrice']
    X.drop([column_to_drop for column_to_drop in columns_to_drop if column_to_drop in X.columns],
           inplace=True,
           axis=1)
    numerical_dataframe = __handle_numerical_data(dataframe=X)
    categorical_dataframe = __handle_categorical_data(dataframe=X, model_dir=model_dir)
    merged_dataframe = numerical_dataframe.join(categorical_dataframe)
    return merged_dataframe


def __get_encoder(model_dir: Path) -> OneHotEncoder:
    encoder_path = Path(model_dir / AppConstant.ENCODER_NAME)
    if encoder_path.exists(): return joblib.load(encoder_path)
    encoder = OneHotEncoder(handle_unknown='ignore', dtype=int, sparse=True)
    joblib.dump(encoder, model_dir / AppConstant.ENCODER_NAME)
    return encoder


def __fill_unknown(columns: List, dataframe) -> pd.DataFrame:
    for column in columns:
        dataframe[column] = dataframe[column].fillna('Unknown')
    return dataframe


def __fill_with_most_frequent(columns: List, dataframe) -> pd.DataFrame:
    for column in columns:
        dataframe[column].fillna(dataframe.value_counts().index[0])
    return dataframe
