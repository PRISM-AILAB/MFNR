import os

import torch
import torch.nn as nn

from src.path import (
    PROCESSED_PATH, 
    SAVE_MODEL_PATH, 
    UTILS_PATH
)
from src.data import DataLoader
from src.utils import (
    save_parquet, 
    load_parquet, 
    load_yaml, 
    set_seed
)
from model.proposed import (
    MFNR, 
    get_data_loader, 
    mfnr_train, 
    mfnr_evaluate
)

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Load Config
    # -------------------------------------------------------------------------
    CONFIG_FPATH = os.path.join(UTILS_PATH, "config.yaml")
    cfg = load_yaml(CONFIG_FPATH)
    dargs = cfg.get("data")
    args = cfg.get("args")

    req_dev  = args.get("device")
    if req_dev == "cuda" and not torch.cuda.is_available():
        print("[INFO] cuda is not available, switched to cpu.")
        use_dev = torch.device("cpu")
    else:
        use_dev = torch.device(req_dev)
    args["device"] = use_dev

    set_seed(cfg.get("seed"))

    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    processed_data_list = set(os.listdir(PROCESSED_PATH))
    TRAIN_FPATH = os.path.join(PROCESSED_PATH, "train.parquet")
    VAL_FPATH = os.path.join(PROCESSED_PATH, "val.parquet")
    TEST_FPATH = os.path.join(PROCESSED_PATH, "test.parquet")

    if {"train.parquet", "val.parquet", "test.parquet"} - processed_data_list:
        dcfg = cfg["data"]
        data_loader = DataLoader(**dargs)
        train, val, test = data_loader.train, data_loader.val, data_loader.test

        save_parquet(train, TRAIN_FPATH)
        save_parquet(val, VAL_FPATH)
        save_parquet(test, TEST_FPATH)

    else: 
        train = load_parquet(TRAIN_FPATH)
        val = load_parquet(VAL_FPATH)
        test = load_parquet(TEST_FPATH)

    args["num_users"] = int(train["user"].nunique())
    args["num_items"] = int(train["item"].nunique())

    train_loader = get_data_loader(args, train, shuffle=True, num_workers=4)
    val_loader = get_data_loader(args, val, shuffle=False, num_workers=4)
    test_loader  = get_data_loader(args, test,  shuffle=False, num_workers=4)

    # -------------------------------------------------------------------------
    # Initialization (model, loss, optimizer)
    # -------------------------------------------------------------------------
    model = MFNR(args).to(args.get("device"))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.get("lr")) # args 추가(다른 optimizer 추가)

    # -------------------------------------------------------------------------
    # Training & Evaluation
    # -------------------------------------------------------------------------
    hist = mfnr_train(args, model, train_loader, val_loader, optimizer, criterion)

    # load best model
    BEST_MODEL_PATH = os.path.join(SAVE_MODEL_PATH, f"{model._get_name()}_bset.pt")
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not found at: {BEST_MODEL_PATH}")
    
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=args.get("device")))
    test_loss, test_preds = mfnr_evaluate(args, model, test_loader, criterion)
    print(f"[TEST] loss={test_loss:.5f}")




