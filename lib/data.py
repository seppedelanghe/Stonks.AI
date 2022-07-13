import numpy as np
import pandas as pd
import torch, os

from torch.utils import data
from sklearn.model_selection import train_test_split

def get_all_csv_files(dir: str):
    files = []
    a = os.listdir(dir)
    for b in a:
        if os.path.isdir(os.path.join(dir, b)):
            c = os.path.join(dir, b, 'csv')
            if os.path.isdir(c):
                files += [os.path.join(c, f) for f in os.listdir(c) if f[-4:] == '.csv']

    return files

def scale(X, y):
    div = 255 / np.max(X)
    X *= div
    y *= div
    return X, y

def make_dataset(df: pd.DataFrame, time_d: int = 10):
    X, y = [], []
    last = []
    for i, row in df.iterrows():
        if row.hasnans:
            if type(last) == type(None):
                continue
            row = last[-1] + (last[-1] - last[len(last) - 2]) / 2


        if type(last) == type(None) or len(last) < time_d:
            last.append(row.values)
            continue
        
        X.append(np.array(last))
        y.append(row.values)
        last.pop(0)
        last.append(row.values)

    return np.array(X), np.array(y)

def get_loader_for_file(path: str, batch_size: int, time_d: int, as_double: bool = False):
    df = pd.read_csv(path)
    df.drop(['Date', 'Volume'], axis=1, inplace=True)
    X, y = make_dataset(df, time_d)
    X, y = scale(X, y)
    if not as_double:
        X, y = X.astype("float32"), y.astype("float32")
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    train_loader = data.DataLoader(data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size)
    test_loader = data.DataLoader(data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=batch_size)

    return train_loader, test_loader
