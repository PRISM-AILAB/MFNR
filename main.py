import os
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

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
    set_seed,
    get_metrics
)
from model.proposed import (
    MFNR, 
    get_data_loader, 
    mfnr_trainer, 
    mfnr_tester
)

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Load Config
    # -------------------------------------------------------------------------
    CONFIG_FPATH = os.path.join(UTILS_PATH, "config.yaml")
    cfg = load_yaml(CONFIG_FPATH)
    dargs = cfg.get("data")
    args = cfg.get("args")
    args["fname"] = dargs.get("fname")

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
    FNAME = dargs.get("fname")
    TRAIN_FPATH = os.path.join(PROCESSED_PATH, f"{FNAME}_train.parquet")
    VAL_FPATH = os.path.join(PROCESSED_PATH, f"{FNAME}_val.parquet")
    TEST_FPATH = os.path.join(PROCESSED_PATH, f"{FNAME}_test.parquet")

    if {f"{FNAME}_train.parquet", f"{FNAME}_val.parquet", f"{FNAME}_test.parquet"} - processed_data_list:
        data_loader = DataLoader(**dargs)
        train, val, test = data_loader.train, data_loader.val, data_loader.test

        save_parquet(train, TRAIN_FPATH)
        save_parquet(val, VAL_FPATH)
        save_parquet(test, TEST_FPATH)

    else: 
        train = load_parquet(TRAIN_FPATH)
        val = load_parquet(VAL_FPATH)
        test = load_parquet(TEST_FPATH)

    total = pd.concat([train, val, test], axis = 0)
    args["num_users"] = int(total["user"].nunique())
    args["num_items"] = int(total["item"].nunique())

    train_loader = get_data_loader(args, train, shuffle=True, num_workers=4)
    val_loader = get_data_loader(args, val, shuffle=False, num_workers=4)
    test_loader  = get_data_loader(args, test,  shuffle=False, num_workers=4)

    # -------------------------------------------------------------------------
    # Initialization (model, loss, optimizer)
    # -------------------------------------------------------------------------
    model = MFNR(args).to(args.get("device"))
    criterion = nn.MSELoss()

    OPTIMIZER = args.get("optimizer").lower()
    if OPTIMIZER == "adam":
        optimizer = optim.Adam(model.parameters(), lr = args.get("lr"))
    elif OPTIMIZER == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr = args.get("lr"))
    elif OPTIMIZER == "sgd":
        optimizer = optim.SGD(model.parameters(), lr = args.get("lr"))
    elif OPTIMIZER == "rmsprop": 
        optimizer = optim.RMSprop(model.parameters(), lr = args.get("lr"))
        
    # -------------------------------------------------------------------------
    # Training & Evaluation
    # -------------------------------------------------------------------------
    hist = mfnr_trainer(args, model, train_loader, val_loader, optimizer, criterion)

    # load best model
    BEST_MODEL_FPATH = os.path.join(SAVE_MODEL_PATH, f"{FNAME}_Best_Model.pth")
    if not os.path.exists(BEST_MODEL_FPATH):
        raise FileNotFoundError(f"Best model not found at: {BEST_MODEL_FPATH}")
    
    model.load_state_dict(torch.load(BEST_MODEL_FPATH, map_location=args.get("device")))
    test_preds, test_trues = mfnr_tester(args, model, test_loader)
    mse, rmse, mae, mape = get_metrics(test_preds, test_trues)
    print(f"[TEST] RMSE={rmse:.5f}  MSE={mse:.5f}  MAE={mae:.5f}  MAPE={mape:.3f}%")





