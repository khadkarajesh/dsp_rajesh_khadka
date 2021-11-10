import os
from pathlib import Path
import pandas as pd

from app.models.inference import predict
from dotenv import load_dotenv

from app.models.preprocess import get_column_transformer, preprocess
from app.models.train import train_model

load_dotenv('.env')

# model_path = Path(os.environ.get('MODEL_PATH'))
# # train model
# train_model(Path(os.environ.get('DATA_PATH'), 'train.csv'), get_column_transformer(model_path), model_path)
#
# # predict model
# predict(pd.read_csv(Path(os.environ.get('DATA_PATH'), 'test.csv')), model_path / 'model.joblib')

preprocess(pd.read_csv(Path(os.environ.get('DATA_PATH'), 'train.csv')), Path("app/models"))
