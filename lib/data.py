import os, gzip, shutil, torch, pickle
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset
from typing import List

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
        return torch.from_numpy(np.array(d))

    def find_ranges(self):
        money = self.X[:, :-1]
        volume = self.X[:, -1:]

        self.scl['money'] = (float(torch.min(money)), float(torch.max(money)))
        self.scl['volume'] = (float(torch.min(volume)), float(torch.max(volume)))

    def scale(self, data: torch.Tensor, new_min=0, new_max=1, with_volume: bool = True):
        money = data[:, :4] if with_volume else data
        money = (money - self.scl['money'][0]) / (self.scl['money'][1] - self.scl['money'][0]) * (new_max - new_min) + new_min
        
        if not with_volume:
            return money

        volume = (data[:, -1:] - self.scl['volume'][0]) / (self.scl['volume'][1] - self.scl['volume'][0]) * (new_max - new_min) + new_min
        return torch.hstack((money, volume))

    def inverse_scale(self, data: torch.Tensor, curr_max = 1, with_vol: bool = False):
        money = data[:, :-1] if with_vol else data
        money = money / curr_max * (self.scl['money'][1] - self.scl['money'][0]) + self.scl['money'][0]

        if not with_vol:
            return money
        
        volume = data[:, -1:] / curr_max * (self.scl['volume'][1] - self.scl['volume'][0]) + self.scl['volume'][0]
        return torch.hstack((money, volume))

    def make_dataset(self, d: List[List[float]]):
        d: torch.Tensor = self.to_tensors(d)
        Xd, yd = None, None

        for i in range(d.shape[0] - self.time_d - 1):
            idx = i + self.time_d
            row = d[idx + 1]
            row = row[:-1] # remove volume is it doesn't need to be predicted
            x = d[i:idx]
            if torch.isnan(torch.sum(x)) or torch.isnan(torch.sum(row)):
                continue # if the row (y) has nans, skip

            if type(Xd) == type(None):
                Xd = x
                yd = row
            else:
                Xd = torch.concat((Xd, x))
                yd = torch.concat((yd, row))
            
        return Xd, yd

    def make_datasets(self, data: dict):
        loop = tqdm(data, total=len(data), leave=False)
        for ticker in loop:
            # to_dataset
            dataset = self.make_dataset(data[ticker])

            # checks
            if type(dataset[0]) == type(None) or type(dataset[1]) == type(None):
                print(f"{ticker} is useless when using time_d {self.time_d}")
                continue

            # add to large dataset
            if type(self.X) == type(None):
                self.X = dataset[0]
                self.y = dataset[1]
            else:
                self.X = torch.concat((self.X, dataset[0]))
                self.y = torch.concat((self.y, dataset[1]))

        # scale
        self.find_ranges()
        self.X = self.scale(self.X)
        self.y = self.scale(self.y, with_volume=False)

    def large_df(self, path):
        files = self.get_all_csv_files(path)
        return pd.concat((pd.read_csv(f) for i, f in tqdm(enumerate(files), total=len(files), leave=False)), ignore_index=True)

    def filter_data(self, df: pd.DataFrame):
        df = df[self.cols].dropna(axis=0)
        return df[df['Low'] < self.max_low]

    def format_file(self, path: str):
        df = pd.read_csv(path)
        df = self.filter_data(df)
        return df.to_numpy().tolist()

    def from_csvs(self, path):
        files = self.get_all_csv_files(path)
        loop = tqdm(files, total=len(files), leave=False)
        return {self.ticker_from_path(path): self.format_file(path) for path in loop}

    def load_datasets(self, Xpath: str, ypath: str, scl: dict):
        assert os.path.isfile(Xpath), 'X file does not exist'
        assert os.path.isfile(ypath), 'y file does not exist'

        self.X: torch.Tensor = pickle.load(open(Xpath, 'rb'))
        self.y: torch.Tensor = pickle.load(open(ypath, 'rb'))

        assert self.X.shape[0] / self.time_d == self.X.shape[0] // self.time_d, 'time_d does not seem to be correct for this dataset'
        assert self.y.shape[0] / self.output_params == self.y.shape[0] // self.output_params , 'the output_params seem to be incorrect for this dataset'
        
        self.X = self.X.reshape((self.X.shape[0] // self.time_d, self.time_d, self.X.shape[-1]))
        self.y = self.y.reshape((self.y.shape[0] // self.output_params, self.output_params))

        self.scl = scl

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