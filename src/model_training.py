"""model training module"""

import torch
from torch import nn as nn

class TabularModel(nn.Module):
    """class for Model"""

    def __init__(self, emb_szs, n_conts, layers, out_sz, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.norm = nn.BatchNorm1d(n_conts)
        self.dropout = nn.Dropout(p)
        n_emb = sum([nf for ni, nf in emb_szs])
        n_in = n_emb + n_conts
        layer_list = []
        
        for i in layers:
            layer_list.append(nn.Linear(n_in, i))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.BatchNorm1d(i))
            layer_list.append(nn.Dropout(p))
            n_in = i
        layer_list.append(nn.Linear(layers[-1], out_sz))
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x_cat, x_cont):
        """forward method for model"""

        embedding = []
        for i, e in enumerate(self.embeds):
            embedding.append(e(x_cat[:, i]))
        x = torch.cat(embedding, 1)
        x = self.dropout(x)

        x_cont = self.norm(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x
            