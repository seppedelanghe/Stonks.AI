from genericpath import isfile
import os, gzip, shutil, torch, pickle, json
import re
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset
from typing import List, Optional, Union
import numpy as np

from pydantic import BaseModel

class StockDatasetConfig(BaseModel):
    ticker: str
    path: str = './'
    time: int = 30

    @property
    def save_path(self):
        return os.path.join(self.path, self.ticker, 'config.json')

    @property
    def Xpath(self):
        return os.path.join(self.path, self.ticker, 'X.bin')
    
    @property
    def ypath(self):
        return os.path.join(self.path, self.ticker, 'y.bin')

class StockDataset(Dataset):
    def __init__(self, config: StockDatasetConfig, X: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None):
        super(StockDataset, self).__init__()

        self.config = config

        self.X = X
        self.y = y

    def save(self):
        os.makedirs(self.config.path, exist_ok=True)

        fx, fy = open(self.config.Xpath, 'wb'), open(self.config.ypath, 'wb')
        
        pickle.dump(self.X, fx)
        pickle.dump(self.y, fy)
        
        fx.close()
        fy.close()

        with open(self.config.save_path, 'w') as f:
            f.write(self.config.json())

        return True

    @classmethod
    def load(cls, config_path: str):
        sd = cls(config=StockDatasetConfig.parse_file(config_path))

        if os.path.isfile(sd.config.Xpath):
            with open(sd.config.Xpath, 'rb') as fx:
                sd.X = pickle.loads(fx)

        if os.path.isfile(sd.config.ypath):
            with open(sd.config.ypath, 'rb') as fy:
                sd.y = pickle.loads(fy)

    def __len__(self):
        return 0 if type(self.X) == type(None) else len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class StonksData:
    def __init__(self, csv_path: str, time: int = 30, output_params: int = 4):
        self.csv_path = csv_path
        self.time = time
        self.output_params = output_params

        self.cols = ['Close']

    '''
        get stock ticker symbol from csv path
    '''
    @staticmethod
    def ticker_from_path(path: str):
        return os.path.basename(path)[:-4]

    '''
        Convert ndarray to tensors
    '''
    @staticmethod
    def to_tensors(d: np.ndarray):
        return torch.from_numpy(d)

    '''
        return min max values for data
    '''
    @staticmethod
    def find_ranges(data: torch.Tensor):
        return float(torch.min(data)), float(torch.max(data))

    '''
        normalize data to values between 0 and 1
    '''
    @staticmethod
    def normalize(data: torch.Tensor):
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

    '''
        inverse normalization
    '''
    @staticmethod
    def inverse_normalize(data: torch.Tensor, new_min: float, new_max: float, curr_max = 1):
        return data / curr_max * (new_max - new_min) + new_min

    @staticmethod
    def make_dataset(data: Union[torch.Tensor, np.ndarray], time: int = 30):        
        if data.shape[0] <= time:
            raise Exception(f"length of data needs to be larger than time. Is {data.shape[0]}, needs to be > {time}")

        if type(data) == np.ndarray:
            data = StonksData.to_tensors(data)

        x = torch.stack([data[i:i + time] for i in range(data.shape[0] - time - 1) if not bool(torch.isnan(torch.sum(data[i:i + time])))]).float()
        y = torch.stack([data[i + time + 1] for i in range(data.shape[0] - time - 1) if not bool(torch.isnan(torch.sum(data[i + time + 1])))]).float()

        return x, y

    def clean_dataframe(self, df: pd.DataFrame):
        return df[self.cols].dropna(axis=0)

    def read_csv(self, file: str, save_dir: str = './'):
        df = self.clean_dataframe(pd.read_csv(file))
        x, y = self.make_dataset(df.to_numpy())
        config = StockDatasetConfig(ticker=self.ticker_from_path(file), path=save_dir, time=self.time)
        return StockDataset(config=config, X=x, y=y)

    def prepare(self, tickers: List[str], save_dir: str):
        csvs = [os.path.join(self.csv_path, f"{tick}.csv") for tick in tickers if os.path.isfile(os.path.join(self.csv_path, f"{tick}.csv"))]
        datasets = [self.read_csv(csv, save_dir) for csv in csvs]
        return [d.config for d in datasets if d.save()]