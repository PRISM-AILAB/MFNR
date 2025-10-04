import os
import numpy as np

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
        y   = torch.tensor(self.labels[idx], dtype=torch.float32)

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
        self.pre_head = nn.Sequential(
            nn.Linear(rating_dim, 64),
            nn.ReLU()
        )
        self.rating_mlp = _MLP(64, n_layer=3, p_drop=0.1)
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
        rep = torch.concat([u_e, u_txt, i_e, i_txt], dim=1)
        h = self.pre_head(rep)
        h = self.rating_mlp(h)
        y = self.out(h)  
        return y
    
# ===============================
# trainer
# ===============================

@torch.no_grad()
def mfnr_evaluate(args, model, data_loader, criterion):
    DEVICE = args.get("device")

    model.eval()
    losses = 0.0
    preds = []
    trues = []

    for batch in data_loader:
        batch = tuple(b.to(DEVICE, non_blocking=True) for b in batch)
        inputs = {
            'uid':          batch[0],
            'iid':          batch[1],
            'user_bert':    batch[2],
            'item_bert':    batch[3],
            'user_roberta': batch[4],
            'item_roberta': batch[5],
        }
        gold_y = batch[6].unsqueeze(-1) if batch[6].dim() == 1 else batch[6]

        pred_y = model(**inputs)
        loss = criterion(pred_y, gold_y)

        losses += loss.item()
        preds.append(pred_y.detach().cpu())
        trues.append(gold_y.detach().cpu())

    losses /= len(data_loader)
    preds = torch.concat(preds, dim=0)
    trues = torch.concat(trues, dim=0) 
    return losses, preds, trues
    

def mfnr_train(args, model, train_loader, valid_loader, optimizer, criterion):
    DEVICE = args.get("device")
    EPOCHS = args.get("num_epochs")
    PATIENCE = args.get("patience")

    train_losses, valid_losses = [], []
    best_loss = float('inf')
    no_improve = 0

    model.to(DEVICE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0

        for batch in train_loader:
            batch = tuple(b.to(DEVICE, non_blocking=True) for b in batch)
            inputs = {
                'uid':          batch[0],
                'iid':          batch[1],
                'user_bert':    batch[2],
                'item_bert':    batch[3],
                'user_roberta': batch[4],
                'item_roberta': batch[5],
            }
            gold_y = batch[6].unsqueeze(-1) if batch[6].dim() == 1 else batch[6]

            optimizer.zero_grad(set_to_none=True)
            pred_y = model(**inputs)
            loss = criterion(pred_y, gold_y)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()

        tr_loss /= len(train_loader)
        train_losses.append(tr_loss)

        val_loss,_ ,_ = mfnr_evaluate(args, model, valid_loader, criterion)
        valid_losses.append(val_loss)

        if epoch % 5 == 0:
            print(f'Epoch: [{epoch}/{EPOCHS}]  Train: {tr_loss:.5f}  Valid: {val_loss:.5f}')

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            if not os.path.exists(SAVE_MODEL_PATH):
                os.makedirs(SAVE_MODEL_PATH)
            SAVE_MODEL_FPATH = os.path.join(SAVE_MODEL_PATH, f'{model._get_name()}_best.pt')
            torch.save(model.state_dict(), SAVE_MODEL_FPATH)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

    return {'train_loss': train_losses, 'valid_loss': valid_losses}