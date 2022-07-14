import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch, os, pickle

from torch.utils import data
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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

def scale(X, scaler_path: str, save: bool = False):
    scl = MinMaxScaler()
    if save:
        scl.fit(X)
        pickle.dump(open(scaler_path, 'wb'))
    else:
        scl = pickle.load(open(scaler_path, 'rb'))

    return scl.transform(X)


def make_dataset(df: np.ndarray, time_d: int = 10):
    X, y = [], []

    for i in range(df.shape[0] - time_d - 1):
        idx = i + time_d
        row = df[idx + 1]
        row = row[:-1] # remove volume is it doesn't need to be predicted
        x = df[i:idx]
        if np.isnan(np.sum(row)) or np.isnan(np.sum(row)):
            continue # if the row (y) has nans, skip

        X.append(x)
        y.append(row)
        
    return np.array(X), np.array(y)

def get_loader_for_file(path: str, batch_size: int, time_d: int, as_double: bool = False):
    df = pd.read_csv(path)
    if len(df) < time_d:
        print(f"This file is useless: {path}")
    
    # remove date tag
    df.drop(['Date'], axis=1, inplace=True)
    
    # scale with min max scaler
    df = pd.DataFrame(scale(df, 'minmax_scaler.bin'), columns=['Low', 'Open', 'Volume', 'High', 'Close', 'Adjusted Close'])
    
    # rearrange columns for 'make_dataset'
    df = df[['Low', 'Open', 'High', 'Close', 'Adjusted Close', 'Volume']]
    
    # make dataset from dataframe
    X, y = make_dataset(df.to_numpy(), time_d)
    
    # if values need to be in floats, convert to float32 instead of float64
    if not as_double:
        X, y = X.astype("float32"), y.astype("float32")
    
    # split in train and test + make pytorch loaders
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    train_loader = data.DataLoader(data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size)
    test_loader = data.DataLoader(data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=batch_size)

    return train_loader, test_loader