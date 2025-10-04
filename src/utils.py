import pandas as pd, numpy as np
import gzip, json, yaml, random, torch
from sklearn.metrics import (
   mean_squared_error, 
   mean_absolute_error, 
   mean_absolute_percentage_error
)


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def load_parquet(fpath):
    return pd.read_parquet(fpath, engine = "pyarrow")

def save_parquet(df: pd.DataFrame, fpath: str):
    df.to_parquet(fpath, engine = "pyarrow")

def load_yaml(fpath: str) -> dict:
   with open(fpath, "r", encoding = "utf-8") as f:
      return yaml.safe_load(f)
   
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_metrics(preds, trues):
    preds = preds.squeeze().numpy() if isinstance(preds, torch.Tensor) else preds
    trues = trues.squeeze().numpy() if isinstance(trues, torch.Tensor) else trues
    
    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues, preds)
    mape = mean_absolute_percentage_error(trues, preds) * 100
    return mse, rmse, mae, mape