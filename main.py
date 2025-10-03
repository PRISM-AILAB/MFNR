import os, argparse

import torch
import torch.nn as nn

from src.path import PROCESSED_PATH, SAVE_MODEL_PATH
from src.data import DataLoader
from src.utils import save_parquet, load_parquet
from model.proposed import get_mfnr_loader, MFNR, mfnr_train, mfnr_evaluate



parser = argparse.ArgumentParser() # 임시 (수정 필요)
parser.add_argument("--model", type=str, required=True, choices=["con", "mul", "attention", "add"])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--chunk_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--data", type=str, default="data")
args = parser.parse_args()

if __name__ == "__main__":
 
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    processed_data_list = os.listdir(PROCESSED_PATH)
    TRAIN_FPATH = os.path.join(PROCESSED_PATH, "train.parquet")
    VAL_FPATH = os.path.join(PROCESSED_PATH, "val.parquet")
    TEST_FPATH = os.path.join(PROCESSED_PATH, "test.parquet")

    if {"train.parquet", "val.parquet", "test.parquet"} - set(processed_data_list):
        FNAME = args.data
        
        data_loader = DataLoader(fname = FNAME)
        train, val, test = data_loader.train, data_loader.val, data_loader.test

        save_parquet(train, TRAIN_FPATH)
        save_parquet(val, VAL_FPATH)
        save_parquet(test, TEST_FPATH)

    else: 
        train = load_parquet(TRAIN_FPATH)
        val = load_parquet(VAL_FPATH)
        test = load_parquet(TEST_FPATH)

    train_loader = get_mfnr_loader(args, train, shuffle=True, num_workers=4)
    val_loader = get_mfnr_loader(args, val, shuffle=False, num_workers=4)
    test_loader  = get_mfnr_loader(args, test,  shuffle=False, num_workers=4)

    # -------------------------------------------------------------------------
    # Initialization (model, loss, optimizer)
    # -------------------------------------------------------------------------
    model = MFNR(args).to(args.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    # -------------------------------------------------------------------------
    # Training & Evaluation
    # -------------------------------------------------------------------------
    hist = mfnr_train(args, model, train_loader, val_loader, optimizer, criterion)

    # save best model
    BEST_MODEL_PATH = os.path.join(SAVE_MODEL_PATH, f"{model._get_name()}_bset.pt")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=args.device))
    test_loss, test_preds = mfnr_evaluate(args, model, test_loader, criterion)
    print(f"[TEST] loss={test_loss:.5f}")




