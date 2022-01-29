from torch import nn
from model import Model_v1, pred,Mydataset,generate_dataloaders
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
from sklearn.metrics import r2_score

rnn_f1 = nn.GRU(1, 16,num_layers=1,dropout=0)
rnn_f3 = nn.GRU(4, 32,num_layers=2,dropout=0)
rnn_f4 = nn.GRU(1, 16,num_layers=1,dropout=0)
embedLayer = nn.Embedding(24*6,16)
mlp = nn.Sequential(
            nn.Linear(16+32+16+16, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
net = Model_v1(rnn_f1,embedLayer, rnn_f3, rnn_f4, mlp)
net.load_state_dict(torch.load("1_29/01_29 16_32.pt"))

train_data = Mydataset("train")
val_data = Mydataset("val")
train_loader, val_loader=generate_dataloaders(train_data, val_data,batch_size=32,isShuffle=False)

train_pred = np.exp(pred(net, train_loader))
train_actl = np.exp(train_data.y)
print(np.sqrt(mean_squared_error(train_actl, train_pred)))
print(r2_score(train_actl, train_pred))
val_pred = pred(net, val_loader)
val_actl = np.exp(val_data.y)
print(np.sqrt(mean_squared_error(val_actl, val_pred)))
print(r2_score(val_actl, val_pred))
