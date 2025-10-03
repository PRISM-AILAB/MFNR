import pandas as pd, numpy as np
import gzip, json, yaml, random, torch

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