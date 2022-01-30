import os

import pandas as pd
import numpy as np
import torch
from torch import nn
from apps.train.model import Model_v1
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# ### Feature engineering

# steps:
# 1. use the train_data set to do feature engineer and get f1, f2, f3, f4
# 2. normalize the f1, f2, f3, f4 by mean_val, log_scale the y (save the mean_val)
# 3. split the train_data to train_set, val_set, test_set

# f1: the same time of last 7 days (extract from y);  shape=(7*1,)
# f2: the datetime as a categorical value (hh:mm); shape=(1,)
# f3: the previous two hour x1, x2, x3, x4 multi-value time series;  shape=(2*6,5)
# f4: the previous two hour y; shape=(2*6,1)

# feed f1, f3, f4 to two different GRU, get o1, o2, o3
# feed f2, o1, o2, o3 to MLP and get final output

def train_test_split(array, test_size):
    num_samples = len(array[0])
    permutation = np.random.permutation(range(num_samples))
    train_split,test_split = [], []
    test_set_size = int(num_samples*test_size)
    for arr in array:
        train_split.append(arr[permutation][:-test_set_size])
        test_split.append(arr[permutation][-test_set_size:])
    return train_split, test_split

def normalize(array,name_array):
    normalize_res=[]
    for arr, name in zip(array,name_array):
        mu, theta =np.mean(arr,axis=0),np.max(arr,axis=0)-np.min(arr,axis=0)
        normalize_res.append((arr-mu)/theta)
        np.save(f'processed/mean_{name}.npy', mu)
        np.save(f"processed/theta_{name}.npy",theta)
    return tuple(normalize_res)

def preprocess(df, X, y, time_point_week):
    # get the f1
    f1=[]
    for i in range(time_point_week,len(df)):
        last_week_demand = []
        for j in range(7):
            last_week_demand.append(y[i-time_point_week+j*24*6])
        f1.append(last_week_demand)
    f1=np.array(f1)

    # get the f2
    time_label=[t[-8:-3] for t in df['DateTime']]
    time2catg=dict(zip(time_label, range(len(set(time_label)))))
    f2=[time2catg[t] for t in time_label[time_point_week:]]
    f2=np.array(f2)

    # get the f3, f4
    f3=[]
    f4=[]
    for i in range(time_point_week,len(df)):
        last_two_hour = X[i-12:i].values
        last_two_hour_demand =y[i-12:i]
        f3.append(last_two_hour)
        f4.append(last_two_hour_demand)
    f3=np.array(f3)
    f4=np.array(f4)
    f1, f3, f4 = normalize([f1, f3, f4],["f1","f3","f4"])

    # save 10% data as test set 
    train_split, test_split = train_test_split(
        [np.log(y[time_point_week:]).values, f1, f2, f3, f4],test_size=0.05)

    for name, arr in zip(["train","val"],[train_split, test_split]):
        for col,i in zip(['y','f1','f2','f3','f4'],arr):
            print(f"{name}_{col}: {i.shape}")
            np.save(f"./processed/{name}_{col}.npy",i)

def prediction(df_all, time2catg, train_size, time_point_week):
    # define the network
    res=[]
    rnn_f1 = nn.GRU(1, 16,num_layers=1,dropout=0)
    rnn_f3 = nn.GRU(4, 32,num_layers=2,dropout=0)
    rnn_f4 = nn.GRU(1, 16,num_layers=1,dropout=0)
    embedLayer = nn.Embedding(24*6,16)
    mlp = nn.Sequential(
                nn.Linear(16+32+16+16, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
    base = os.getcwd()

    net = Model_v1(rnn_f1,embedLayer, rnn_f3, rnn_f4, mlp)
    net.load_state_dict(torch.load(base + "/apps/train/1_29/01_30 00_26.pt"))
    net.eval()
    df_all_X=df_all[['X1','X2','X3','X4']]

    mean_f1 = np.load(base + "/apps/train/processed/mean_f1.npy")
    mean_f3 = np.load(base + "/apps/train/processed/mean_f3.npy")
    mean_f4 = np.load(base + "/apps/train/processed/mean_f4.npy")
    theta_f1 = np.load(base + "/apps/train/processed/theta_f1.npy")
    theta_f3 = np.load(base + "/apps/train/processed/theta_f3.npy")
    theta_f4 = np.load(base + "/apps/train/processed/theta_f4.npy")
    for idx in range(train_size,len(df_all)):
        # get the f1
        last_week_demand = []
        for j in range(7):
            last_week_demand.append(df_all['Y'][idx-time_point_week+j*24*6])
        f1=(np.array([last_week_demand])-mean_f1)/theta_f1
        # get the f2
        f2=[time2catg[df_all['DateTime'][idx][-8:-3]]]
        f2=np.array(f2)
        # get the f3, f4
        last_two_hour = df_all_X[idx-12:idx].values
        last_two_hour_demand =df_all['Y'][idx-12:idx]
        f3=(np.array([last_two_hour])-mean_f3)/theta_f3
        f4=np.array(([last_two_hour_demand])-mean_f4)/theta_f4
        # reshape the value
        f1=torch.tensor(f1).unsqueeze(-1)
        f2=torch.tensor(f2,dtype=torch.long)
        f3=torch.tensor(f3)
        f4=torch.tensor(f4).unsqueeze(-1)
        # do the prediction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        net.to(device)
        net.eval()
        f1=f1.to(device)
        f2=f2.to(device)
        f3=f3.to(device)
        f4=f4.to(device)
        y_hat=net(f1,f2,f3,f4)
        y_pred=np.exp(y_hat.cpu().detach().numpy()[0])
        df_all.iat[idx,-1]=y_pred
        res.append(y_pred)
    return res

def api(train_path, test_path):
    df=pd.read_csv(train_path)
    print(f"the length of the dataframe is {len(df)}")
    # X, y = df[['X1','X2','X3','X4']],df['Y']
    time_point_week=7*24*6
    # preprocess(df,X, y,time_point_week)

    df_test=pd.read_csv(test_path)
    # # get the dict
    time_label=[t[-8:-3] for t in df['DateTime'][:400]]
    time2catg=dict(zip(time_label, range(len(set(time_label)))))

    # # do the prediction
    train_size=len(df)
    df_all = df.append(df_test).reset_index(drop=True)
    res = prediction(df_all, time2catg, train_size, time_point_week)

    return res