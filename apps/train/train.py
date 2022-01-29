from torch import nn
from model import Mydataset, generate_dataloaders, Model_v1, train_model
from argparse import Namespace

# Net
# f1: the same time of last 7 days (extract from y);  shape=(7*1,)
# f2: the datetime as a categorical value (hh:mm); shape=(1,)
# f3: the previous two hour x1, x2, x3, x4 multi-value time series;  shape=(2*6,5)
# f4: the previous two hour y; shape=(2*6,1)

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

# Dataloader
train_data = Mydataset("train")
val_data = Mydataset("val")
train_loader, val_loader=generate_dataloaders(train_data, val_data,batch_size=32)

# confg
my_confg={"load_from_path":None,"save_to_path":"1_29/","lr":0.001,"weight_decay":0,"num_epochs":50}
my_confg = Namespace(**my_confg)

# Train
train_model(net,train_loader, val_loader, my_confg)
