import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch, os, pickle

from torch.utils import data
from tqdm import tqdm
from cachetools import cached, Cache

DEFAULT_DATA_PATH = './stock_market_data'
SCALER_PATH = './scalers/minmax_scaler.bin'
VOLUME_SCALER_PATH = './scalers/volume_scaler.bin'
CACHE = Cache(maxsize=1000)


MONEY_RANGE = [0.0, 8324640145408.0]
ADJUSTED_RANGE = [-616456459764173628137646621458432.0, 942168706172424929960592932864.0]
VOLUME_RANGE = [0.0 ,7421640800.0]

def scale(data: torch.Tensor, new_min=0, new_max=1):
    money = data[:, :4]
    adj = data[:, 4:5]
    volume = data[:, -1:]

    money = (money - MONEY_RANGE[0]) / (MONEY_RANGE[1] - MONEY_RANGE[0]) * (new_max - new_min) + new_min
    adj = (adj - ADJUSTED_RANGE[0]) / (ADJUSTED_RANGE[1] - ADJUSTED_RANGE[0]) * (new_max - new_min) + new_min
    volume = (volume - VOLUME_RANGE[0]) / (VOLUME_RANGE[1] - VOLUME_RANGE[0]) * (new_max - new_min) + new_min
    return torch.hstack((money, adj, volume))

def inverse_scale(data: torch.Tensor, curr_max=1, with_vol: bool = False):
    money = data[:, :4]
    adj = data[:, 4:5]

    money = money / curr_max * (MONEY_RANGE[1] - MONEY_RANGE[0]) + MONEY_RANGE[0]
    adj = adj / curr_max * (ADJUSTED_RANGE[1] - ADJUSTED_RANGE[0]) + ADJUSTED_RANGE[0]

    if with_vol:
        volume = data[:, -1:]
        volume = volume / curr_max * (VOLUME_RANGE[1] - VOLUME_RANGE[0]) + VOLUME_RANGE[0]

        return torch.hstack((money, adj, volume))
    return torch.hstack((money, adj))



def get_all_csv_files(dir: str):
    files = []
    a = os.listdir(dir)
    for b in a:
        if os.path.isdir(os.path.join(dir, b)):
            c = os.path.join(dir, b, 'csv')
            if os.path.isdir(c):
                files += [os.path.join(c, f) for f in os.listdir(c) if f[-4:] == '.csv']

    return files

def large_df(files):
    return pd.concat((pd.read_csv(f) for i, f in tqdm(enumerate(files), total=len(files))), ignore_index=True)

# def make_scaler():
#     print('=> Making scaler from all files')
#     x = large_df(get_all_csv_files(DEFAULT_DATA_PATH))
#     x = x.dropna(axis=0)
#     x = x.drop(['Date'], axis=1)
#     x = x[['Low', 'Open', 'High', 'Close', 'Adjusted Close', 'Volume']]
#     x = x.to_numpy()
    
#     scl = MinMaxScaler(feature_range=(0, 1))
#     vscl = MinMaxScaler(feature_range=(0, 1))

#     scl = scl.fit(x[:, :-1])
#     vscl = vscl.fit(x[:, -1:])

#     pickle.dump(scl, open(SCALER_PATH, 'wb'))
#     pickle.dump(vscl, open(VOLUME_SCALER_PATH, 'wb'))

# def inverse_scale(data, with_volume=True):
#     if with_volume:
#         scl: MinMaxScaler = pickle.load(open(SCALER_PATH, 'rb'))
#         vscl: MinMaxScaler = pickle.load(open(VOLUME_SCALER_PATH, 'rb'))
#         data[:, :-1] = scl.inverse_transform(data[:, :-1])
#         data[:, -1:] = vscl.inverse_transform(data[:, -1:])
#         return data
    
#     scl: MinMaxScaler = pickle.load(open(SCALER_PATH, 'rb'))
#     return scl.inverse_transform(data)


# def scale(x):
#     if not os.path.isfile(SCALER_PATH) or not os.path.isfile(SCALER_PATH):
#         make_scaler()
    
#     scl: MinMaxScaler = pickle.load(open(SCALER_PATH, 'rb'))
#     vscl: MinMaxScaler = pickle.load(open(VOLUME_SCALER_PATH, 'rb'))
#     x[:, :-1] = scl.transform(x[:, :-1])
#     x[:, -1:] = vscl.transform(x[:, -1:])
#     return x

def make_dataset(df: np.ndarray, time_d: int = 10):
    X, y = [], []

    for i in range(df.shape[0] - time_d - 1):
        idx = i + time_d
        row = df[idx + 1]
        row = row[:-1] # remove volume is it doesn't need to be predicted
        x = df[i:idx]
        if np.isnan(np.sum(x)) or np.isnan(np.sum(row)):
            continue # if the row (y) has nans, skip

        X.append(x)
        y.append(row)
        
    return np.array(X), np.array(y)

@cached(cache=CACHE)
def get_loader_for_file(path: str, batch_size: int, time_d: int, as_double: bool = False):
    df = pd.read_csv(path)
    if len(df) < time_d:
        print(f"This file is useless: {path}")
    
    # remove date tag
    df.drop(['Date'], axis=1, inplace=True)
    df = df[['Low', 'Open', 'High', 'Close', 'Adjusted Close', 'Volume']]
    
    # scale with min max scaler
    df = pd.DataFrame(scale(torch.tensor(df.to_numpy())), columns=['Low', 'Open', 'High', 'Close', 'Adjusted Close', 'Volume'])
    
    # rearrange columns for 'make_dataset'
    
    # make dataset from dataframe
    X, y = make_dataset(df.to_numpy(), time_d)
    
    # if values need to be in floats, convert to float32 instead of float64
    if not as_double:
        X, y = X.astype("float32"), y.astype("float32")
    
    # split in train and test + make pytorch loaders
    train_loader = data.DataLoader(data.TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=batch_size)

    return train_loader