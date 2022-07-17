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
CACHE = Cache(maxsize=1100)

MONEY_RANGE = [0.0, 8324640145408.0]
ADJUSTED_RANGE = [-616456459764173628137646621458432.0, 942168706172424929960592932864.0]
VOLUME_RANGE = [0.0 ,7421640800.0]

def scale(data: torch.Tensor, new_min=0, new_max=1):
    money = data[:, :4]
    adj = data[:, 4:5]
    volume = data[:, -1:]

    money_range = (torch.min(money), torch.max(money))
    adj_range = (torch.min(adj), torch.max(adj))
    volume_range = (torch.min(volume), torch.max(volume))

    money = (money - money_range[0]) / (money_range[1] - money_range[0]) * (new_max - new_min) + new_min
    adj = (adj - adj_range[0]) / (adj_range[1] - adj_range[0]) * (new_max - new_min) + new_min
    volume = (volume - volume_range[0]) / (volume_range[1] - volume_range[0]) * (new_max - new_min) + new_min
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
    # rearrange columns for 'make_dataset'
    df = df[['Low', 'Open', 'High', 'Close', 'Adjusted Close', 'Volume']]
    
    # scale with min max scaler
    df = pd.DataFrame(scale(torch.tensor(df.to_numpy())), columns=['Low', 'Open', 'High', 'Close', 'Adjusted Close', 'Volume'])
    
    # make dataset from dataframe
    X, y = make_dataset(df.to_numpy(), time_d)
    
    # if values need to be in floats, convert to float32 instead of float64
    if not as_double:
        X, y = X.astype("float32"), y.astype("float32")
    
    # split in train and test + make pytorch loaders
    train_loader = data.DataLoader(data.TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=batch_size)

    return train_loader




from torch.utils.data import Dataset

class PreprocessedDataset(Dataset):
    def __init__(self, Xpath, ypath, time_d: int = 10, output_params: int = 4):
        assert os.path.isfile(Xpath), 'X file does not exist'
        assert os.path.isfile(ypath), 'y file does not exist'

        self.X: torch.Tensor = pickle.load(open(Xpath, 'rb'))
        self.y: torch.Tensor = pickle.load(open(ypath, 'rb'))

        assert self.X.shape[0] / time_d == self.X.shape[0] // time_d, 'time_d does not seem to be correct for this dataset'
        assert self.y.shape[0] / output_params == self.y.shape[0] // output_params , 'the output_params seem to be incorrect for this dataset'
        
        self.X = self.X.reshape((self.X.shape[0] // time_d, time_d, self.X.shape[-1]))
        self.y = self.y.reshape((self.y.shape[0] // output_params, output_params))

    def __len__(self):
        return self.X.shape[0] - 1

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def set_scale(self, mx: float, mx_vol: float):
        self.min = 0.0
        self.max = mx
        self.min_vol = 0.0
        self.max_vol = mx_vol

    def inverse_scale(self, data: torch.Tensor, curr_max = 1, with_vol: bool = False):
        money = data[:, :-1] if with_vol else data
        money = money / curr_max * (self.max - self.min) + self.min

        if not with_vol:
            return money
        
        volume = data[:, -1:] / curr_max * (self.max_vol - self.min_vol) + self.min_vol
        return torch.hstack((money, volume))