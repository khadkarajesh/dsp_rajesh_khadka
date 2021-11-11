import os
from pathlib import Path

from dotenv import load_dotenv

from app.models.inference import predict
from app.models.train import train_model

load_dotenv('.env')

# model_path = Path(os.environ.get('MODEL_PATH'))
# # train model
# train_model(Path(os.environ.get('DATA_PATH'), 'train.csv'), get_column_transformer(model_path), model_path)
#
# # predict model
# predict(pd.read_csv(Path(os.environ.get('DATA_PATH'), 'test.csv')), model_path / 'model.joblib')

print(train_model("data/house-prices/train.csv", Path("app/models")))
# print(predict(Path(os.environ.get('DATA_PATH'), 'train.csv'), Path("app/models")))
