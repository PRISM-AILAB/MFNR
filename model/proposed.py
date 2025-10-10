import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.path import SAVE_MODEL_PATH

# ===============================
# Dataset
# ===============================
class MFNRDataset(Dataset):
    def __init__(self, df):
        self.user_id = df.user.values
        self.item_id = df.item.values
        self.labels  = df.rating.values.astype(np.float32)

        self.user_bert    = torch.tensor(np.stack(df.user_bert.to_list()), dtype=torch.float32)
        self.item_bert    = torch.tensor(np.stack(df.item_bert.to_list()), dtype=torch.float32)
        self.user_roberta = torch.tensor(np.stack(df.user_roberta.to_list()), dtype=torch.float32)
        self.item_roberta = torch.tensor(np.stack(df.item_roberta.to_list()), dtype=torch.float32)

    def __len__(self):
        return len(self.user_id)

    def __getitem__(self, idx):
        uid = torch.tensor(self.user_id[idx], dtype=torch.long)   
        iid = torch.tensor(self.item_id[idx], dtype=torch.long)
        y   = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(-1)

        u_b = self.user_bert[idx]       
        i_b = self.item_bert[idx]      
        u_r = self.user_roberta[idx]    
        i_r = self.item_roberta[idx]    

        return uid, iid, u_b, i_b, u_r, i_r, y

def get_data_loader(args, df, shuffle, num_workers):
    BATCH_SIZE = args.get("batch_size")
    dset = MFNRDataset(df)
    return DataLoader(dset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=num_workers, drop_last=False)

# ===============================
# Model
# ===============================
class _MLP(nn.Module):
    def __init__(self, first_dim: int, n_layer: int, p_drop: float = 0.1):
        super().__init__()
        layers = []
        dim = first_dim
        for _ in range(n_layer):
            next_dim = max(dim // 2, 1)  
            layers += [
                nn.Linear(dim, next_dim),  
                nn.ReLU(), 
                nn.Dropout(p_drop)
            ]
            dim = next_dim  
        self.net = nn.Sequential(*layers)
        self.out_dim = dim if n_layer > 0 else first_dim

    def forward(self, x):
        return self.net(x)
    
class MFNR(nn.Module):
    def __init__(self, args):
        super().__init__()
        K = args.get("latent_dim")
        NUM_USERS = args.get("num_users")
        NUM_ITEMS = args.get("num_items")

        # ID embedding
        self.user_emb = nn.Embedding(NUM_USERS, K)
        self.item_emb = nn.Embedding(NUM_ITEMS, K)

        # text input + MLP
        self.user_mlp  = _MLP(1536, n_layer=4, p_drop=0.1)
        self.item_mlp  = _MLP(1536, n_layer=4, p_drop=0.1)

        # rating MLP
        rating_dim =  K * 2 + self.user_mlp.out_dim + self.item_mlp.out_dim
        self.rating_mlp = _MLP(rating_dim, n_layer=3, p_drop=0.1)
        self.out = nn.Linear(self.rating_mlp.out_dim, 1)

    def forward(self, uid, iid, user_bert, item_bert, user_roberta, item_roberta):
        # ID embedding
        u_e = self.user_emb(uid)      
        i_e = self.item_emb(iid)      

        # bert + roberta
        u_txt = torch.concat([user_bert, user_roberta], dim=1)   
        i_txt = torch.concat([item_bert, item_roberta], dim=1)   

        # text MLP
        u_txt = self.user_mlp(u_txt)
        i_txt = self.item_mlp(i_txt)

        # rating MLP
        x = torch.concat([u_e, u_txt, i_e, i_txt], dim=1)
        x = self.rating_mlp(x)
        y = self.out(x)  
        return y
    
# ===============================
# trainer
# ===============================
def mfnr_trainer(args, model, train_loader, valid_loader, optimizer, criterion):
    """
    Trainer with tqdm bars & best checkpoint.
    Expects each batch as:
      (uid, iid, user_bert, item_bert, user_roberta, item_roberta, outputs)
    """
    FNAME  = args.get("fname")
    DEVICE = args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = args.get("num_epochs", 10)

    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    best_loss = float("inf")
    model.to(DEVICE)

    for epoch in range(1, EPOCHS + 1):
        # ========== TRAIN ==========
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"[Epoch {epoch}] Training", leave=False):
            (uid, iid,
             user_bert, item_bert,
             user_roberta, item_roberta,
             outputs) = batch

            uid           = uid.to(DEVICE)
            iid           = iid.to(DEVICE)
            user_bert     = user_bert.to(DEVICE)
            item_bert     = item_bert.to(DEVICE)
            user_roberta  = user_roberta.to(DEVICE)
            item_roberta  = item_roberta.to(DEVICE)
            outputs       = outputs.to(DEVICE)

            optimizer.zero_grad()
            pred_y = model(uid, iid, user_bert, item_bert, user_roberta, item_roberta)
            loss = criterion(pred_y, outputs)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # ========== VALID ==========
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"[Epoch {epoch}] Validation", leave=False):
                (uid, iid,
                 user_bert, item_bert,
                 user_roberta, item_roberta,
                 outputs) = batch

                uid           = uid.to(DEVICE)
                iid           = iid.to(DEVICE)
                user_bert     = user_bert.to(DEVICE)
                item_bert     = item_bert.to(DEVICE)
                user_roberta  = user_roberta.to(DEVICE)
                item_roberta  = item_roberta.to(DEVICE)
                outputs       = outputs.to(DEVICE)

                pred_y = model(uid, iid, user_bert, item_bert, user_roberta, item_roberta)
                val_loss = criterion(pred_y, outputs)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / max(1, len(valid_loader))

        # ========== SAVE BEST ==========
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            fpath = os.path.join(SAVE_MODEL_PATH, f"{FNAME}_Best_Model.pth")
            torch.save(model.state_dict(), fpath)
            print(f"New best model saved at Epoch {epoch} (Val Loss: {avg_val_loss:.4f})")

        # ========== LOG ==========
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


def mfnr_tester(args, model, test_loader):
    """
    Tester returning numpy arrays (preds, trues).
    Expects each batch as:
      (uid, iid, user_bert, item_bert, user_roberta, item_roberta, outputs)
    """
    DEVICE = args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE).eval()

    preds, trues = [], []

    for batch in test_loader:
        (uid, iid,
         user_bert, item_bert,
         user_roberta, item_roberta,
         outputs) = batch

        uid           = uid.to(DEVICE)
        iid           = iid.to(DEVICE)
        user_bert     = user_bert.to(DEVICE)
        item_bert     = item_bert.to(DEVICE)
        user_roberta  = user_roberta.to(DEVICE)
        item_roberta  = item_roberta.to(DEVICE)
        outputs       = outputs.to(DEVICE)

        pred_y = model(uid, iid, user_bert, item_bert, user_roberta, item_roberta)

        preds.append(pred_y.detach().cpu())
        trues.append(outputs.detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    return preds, trues
