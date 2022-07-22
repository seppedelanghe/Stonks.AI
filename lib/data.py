import os, gzip, shutil, torch, pickle, json
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset
from typing import List
import numpy as np

class StonksDataset(Dataset):
    def __init__(self, time_d: int = 10, output_params: int = 4):
        self.time_d = time_d
        self.output_params = output_params

        self.max_low = 5e3
        self.cols = ['Low', 'Open', 'High', 'Close', 'Volume']
        self.compressed = False

        self.X = None
        self.y = None

        self.scl = {
            'money': (0.0, 1.0),
            'volume': (0.0, 1.0),
        }

    def __len__(self):
        return 0 if type(self.X) == type(None) else len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @staticmethod
    def ticker_from_path(path: str):
        return os.path.basename(path)[:-4]

    @staticmethod
    def get_all_csv_files(dir: str):
        files = []
        a = os.listdir(dir)
        for b in a:
            if os.path.isdir(os.path.join(dir, b)):
                c = os.path.join(dir, b, 'csv')
                if os.path.isdir(c):
                    files += [os.path.join(c, f) for f in os.listdir(c) if f[-4:] == '.csv']

        return files

    @staticmethod
    def to_tensors(d):
        return torch.from_numpy(d)

    def find_ranges(self):
        money = self.X[:, :-1]
        volume = self.X[:, -1:]

        self.scl['money'] = (float(torch.min(money)), float(torch.max(money)))
        self.scl['volume'] = (float(torch.min(volume)), float(torch.max(volume)))

    def scale(self, data: torch.Tensor, new_min=0, new_max=1, with_volume: bool = True):
        money = (data[:, :, :4] if with_volume else data - self.scl['money'][0]) / (self.scl['money'][1] - self.scl['money'][0]) * (new_max - new_min) + new_min
        
        if not with_volume:
            return money

        volume = (data[:, :, -1:] - self.scl['volume'][0]) / (self.scl['volume'][1] - self.scl['volume'][0]) * (new_max - new_min) + new_min
        data[:, :, :4] = money
        data[:, :, -1:] = volume
        return data

    def inverse_scale(self, data: torch.Tensor, curr_max = 1, with_vol: bool = False):
        money = data[:, :-1] if with_vol else data
        money = money / curr_max * (self.scl['money'][1] - self.scl['money'][0]) + self.scl['money'][0]

        if not with_vol:
            return money
        
        volume = data[:, -1:] / curr_max * (self.scl['volume'][1] - self.scl['volume'][0]) + self.scl['volume'][0]
        return torch.hstack((money, volume))

    def make_dataset(self, d: List[List[float]]):
        d: torch.Tensor = self.to_tensors(d)
        x = torch.stack([d[i:i + self.time_d] for i in range(d.shape[0] - self.time_d - 1) if not torch.isnan(torch.sum(d[i:i + self.time_d]))])
        rows = torch.stack([d[i + self.time_d + 1][:-1] for i in range(d.shape[0] - self.time_d - 1) if not torch.isnan(torch.sum(d[i + self.time_d + 1][:-1]))])

        if type(self.X) == type(None):
            self.X = x
            self.y = rows
        else:
            self.X = torch.concat((self.X, x))
            self.y = torch.concat((self.y, rows))
        
    def make_datasets(self, data: dict):
        [self.make_dataset(data[ticker]) for ticker in tqdm(data, total=len(data), leave=False)]

        # scale
        self.find_ranges()
        self.X = self.scale(self.X).float()
        self.y = self.scale(self.y, with_volume=False).float()

    def large_df(self, path):
        files = self.get_all_csv_files(path)
        return pd.concat((pd.read_csv(f) for i, f in tqdm(enumerate(files), total=len(files), leave=False)), ignore_index=True)

    def filter_data(self, df: pd.DataFrame):
        return df[self.cols][df['Low'] < self.max_low].dropna(axis=0)

    def format_file(self, path: str):
        df = pd.read_csv(path)
        df = self.filter_data(df)
        return df.to_numpy()

    def from_csvs(self, path):
        files = self.get_all_csv_files(path)
        loop = tqdm(files, total=len(files), leave=False)
        return {self.ticker_from_path(path): self.format_file(path) for path in loop}

    def save_datasets(self, folder: str):
        pickle.dump(self.X, open(os.path.join(folder, f"{self.time_d}_{self.output_params}_X.bin"), 'wb'))
        pickle.dump(self.y, open(os.path.join(folder, f"{self.time_d}_{self.output_params}_y.bin"), 'wb'))

        config = {
            'X': os.path.join(folder, f"{self.time_d}_{self.output_params}_X.bin"),
            'y': os.path.join(folder, f"{self.time_d}_{self.output_params}_y.bin"),
            'time_d': self.time_d,
            'outputs': self.output_params,
            'scl': self.scl,
            'low_max': self.max_low
        }

        with open(os.path.join(folder, f"{self.time_d}_{self.output_params}_config.json"), 'w') as f:
            f.write(json.dumps(config))

    def load_datasets(self, config_path: str):
        with open(config_path, 'r') as f:
            config = json.loads(f.read())

        self.time_d = config['time_d']
        self.output_params = config['outputs']
        self.scl = config['scl']
        self.max_low = config['low_max']

        assert os.path.isfile(config['X']), 'X file does not exist'
        assert os.path.isfile(config['y']), 'y file does not exist'

        self.X: torch.Tensor = pickle.load(open(config['X'], 'rb'))
        self.y: torch.Tensor = pickle.load(open(config['y'], 'rb'))

    def save_data(self, data: dict, path: str):
        if os.path.isfile(path):
            raise Exception('File already exists')

        pickle.dump(data, open(path, 'wb'))

        if self.compressed:
            with open(path, 'rb') as f_in:
                with gzip.open(path+'.gz', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    def load_data(self, path: str):
        if not os.path.isfile(path):
            raise Exception('File does not exist')

        data = None
        if self.compressed:
            with gzip.open(path, 'rb') as f_in:
                with open(path.replace('.gz', ''), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        return pickle.load(open(path.replace('.gz', '') if self.compressed else path, 'rb'))