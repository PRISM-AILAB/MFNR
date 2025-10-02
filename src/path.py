import os

BASE_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

DATA_PATH = os.path.join(BASE_PATH, "data")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed")
RAW_PATH = os.path.join(DATA_PATH, "raw")

UTILS_PATH = os.path.join(BASE_PATH, "src")