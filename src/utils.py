import pandas as pd
import gzip, json

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