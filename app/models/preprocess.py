from pathlib import Path
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def get_column_transformer() -> ColumnTransformer:
    numerical_preprocessor = Pipeline(steps=[('imputer', SimpleImputer()),
                                             ('scaler', MinMaxScaler())])
    categorical_preprocessor = Pipeline(
        steps=[('encoder', OneHotEncoder(handle_unknown="ignore"))])
    column_transformer = ColumnTransformer(
        transformers=[('num', numerical_preprocessor, selector(dtype_exclude="object")),
                      ('cat', categorical_preprocessor, selector(dtype_exclude="int64"))],
        remainder='passthrough')
    return column_transformer


def fill_unknown(columns: List, dataframe) -> pd.DataFrame:
    for column in columns:
        dataframe[column].fillna('Unknown')
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


path = Path("../../data/house-prices", "train.csv")
get_column_transformer()
# print(q)
