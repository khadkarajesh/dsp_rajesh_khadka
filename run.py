from pathlib import Path
import pandas as pd

from app.models.inference import predict

model_path = Path('app/models/', 'model.joblib')
data_path = Path("data/house-prices/test.csv")
print(predict(pd.read_csv(data_path), model_path))
