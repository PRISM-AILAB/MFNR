import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Dataset
# ===============================
class MFNRDataset(Dataset):
    def __init__(self, dataframe):
        self.user_id = dataframe.user.values
        self.item_id = dataframe.item.values
        self.labels  = dataframe.rating.values.astype(np.float32)

        self.user_bert    = torch.tensor(np.stack(dataframe.user_bert.to_list()), dtype=torch.float32)
        self.item_bert    = torch.tensor(np.stack(dataframe.item_bert.to_list()), dtype=torch.float32)
        self.user_roberta = torch.tensor(np.stack(dataframe.user_roberta.to_list()), dtype=torch.float32)
        self.item_roberta = torch.tensor(np.stack(dataframe.item_roberta.to_list()), dtype=torch.float32)

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

# ===============================
# Model
# ===============================
class _MLP(nn.Module):
    def __init__(self, first_dim: int, n_layer: int, p_drop: float = 0.1):
        super().__init__()
        layers = []
        dim = first_dim
        for _ in range(n_layer):
            layers += [
                nn.Linear(dim, dim), 
                nn.ReLU(), 
                nn.Dropout(p_drop)
            ]
            dim = max(dim // 2, 1)
        self.net = nn.Sequential(*layers)
        self.out_dim = dim if n_layer > 0 else first_dim

    def forward(self, x):
        return self.net(x)
    
class MFNR(nn.Module):
    def __init__(self, args):
        super().__init__()
        k = args.latent_dim

        # ID embedding
        self.user_emb = nn.Embedding(args.num_users, k)
        self.item_emb = nn.Embedding(args.num_items, k)

        # text input + MLP
        self.user_proj = nn.Linear(args.user_txt_dim, 512)
        self.item_proj = nn.Linear(args.item_txt_dim, 512)
        self.user_mlp  = _MLP(512, n_layer=4, p_drop=0.1)
        self.item_mlp  = _MLP(512, n_layer=4, p_drop=0.1)

        # rating MLP
        rating_dim = k + self.user_mlp.out_dim + k + self.item_mlp.out_dim
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
        u_txt = self.user_mlp(torch.relu(self.user_proj(u_txt)))
        i_txt = self.item_mlp(torch.relu(self.item_proj(i_txt)))

        # rating MLP
        rep = torch.concat([u_e, u_txt, i_e, i_txt], dim=1)
        h = self.pre_head(rep)
        h = self.rating_mlp(h)
        y = self.out(h)  
        return y