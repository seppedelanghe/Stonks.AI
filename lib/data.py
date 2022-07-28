import os, torch, pickle

import pandas as pd
import numpy as np
import pandas_datareader as pdr

from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Union
from pydantic import BaseModel


class StockDatasetConfig(BaseModel):
    ticker: str
    time: int = 30
    base_dir: str = './'
    ranges: Optional[Tuple[float, float]] = None

    @property
    def path(self):
        return os.path.join(self.base_dir, self.ticker)

    @property
    def config_path(self):
        return os.path.join(self.path, 'config.json')

    @property
    def Xpath(self):
        return os.path.join(self.path, 'X.bin')
    
    @property
    def ypath(self):
        return os.path.join(self.path, 'y.bin')

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

        with open(self.config.config_path, 'w') as f:
            f.write(self.config.json())

        return True

    @classmethod
    def load(cls, config_path: str):
        sd = cls(config=StockDatasetConfig.parse_file(config_path))

        if os.path.isfile(sd.config.Xpath):
            with open(sd.config.Xpath, 'rb') as fx:
                sd.X = pickle.loads(fx.read())

        if os.path.isfile(sd.config.ypath):
            with open(sd.config.ypath, 'rb') as fy:
                sd.y = pickle.loads(fy.read())

        return sd

    def __len__(self):
        return 0 if type(self.X) == type(None) else len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class StonksData:
    def __init__(self, time: int = 30, cols = ["Close"], prediction_mode: bool = False):
        self.time = time
        self.prediction_mode = prediction_mode
        self.cols = cols

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
    def find_ranges(data: Union[torch.Tensor, np.ndarray]):
        if type(data) == np.ndarray:
            data = StonksData.to_tensors(data)
        return float(torch.min(data)), float(torch.max(data))

    '''
        normalize data to values between 0 and 1
    '''
    @staticmethod
    def normalize(data: Union[torch.Tensor, np.ndarray]):
        if type(data) == np.ndarray:
            data = StonksData.to_tensors(data)
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data)), (float(torch.min(data)), float(torch.max(data)))

    '''
        inverse normalization
    '''
    @staticmethod
    def inverse_normalize(data: Union[torch.Tensor, np.ndarray], new_min: float, new_max: float, curr_max = 1):
        if type(data) == np.ndarray:
            data = StonksData.to_tensors(data)
        return data / curr_max * (new_max - new_min) + new_min

    def make_dataset(self, data: Union[torch.Tensor, np.ndarray], time: int = 30):        
        if data.shape[0] <= time:
            raise Exception(f"length of data needs to be larger than time. Is {data.shape[0]}, needs to be > {time}")

        if type(data) == np.ndarray:
            data = StonksData.to_tensors(data)

        x_range = data.shape[0] - time + 1 if self.prediction_mode else data.shape[0] - time
        x = torch.stack([data[i:i + time] for i in range(x_range) if not bool(torch.isnan(torch.sum(data[i:i + time])))]).float()
        if self.prediction_mode:
            return x

        y = torch.stack([data[i + time] for i in range(data.shape[0] - time) if not bool(torch.isnan(torch.sum(data[i + time])))]).float()

        return x, y

    def clean_dataframe(self, df: pd.DataFrame):
        return df[self.cols].dropna(axis=0)

    def _handle_df(self, df: pd.DataFrame):
        df = self.clean_dataframe(df)
        data, ranges = self.normalize(df.to_numpy())
        return self.make_dataset(data), ranges

    def read_csv(self, file: str, save_dir: str = './'):
        if self.prediction_mode:
            x, ranges = self._handle_df(pd.read_csv(file))
            config = StockDatasetConfig(ticker=self.ticker_from_path(file), base_dir=save_dir, time=self.time, ranges=ranges)
            return StockDataset(config=config, X=x)

        (x, y), ranges = self._handle_df(pd.read_csv(file))
        config = StockDatasetConfig(ticker=self.ticker_from_path(file), base_dir=save_dir, time=self.time, ranges=ranges)
        return StockDataset(config=config, X=x, y=y)

    def from_yahoo(self, ticker: str, save_dir: str = './'):
        if self.prediction_mode:
            x, ranges = self._handle_df(pdr.get_data_yahoo(ticker))
            config = StockDatasetConfig(ticker=ticker, base_dir=save_dir, time=self.time, ranges=ranges)
            return StockDataset(config=config, X=x)

        (x, y), ranges = self._handle_df(pdr.get_data_yahoo(ticker))
        config = StockDatasetConfig(ticker=ticker, base_dir=save_dir, time=self.time, ranges=ranges)
        return StockDataset(config=config, X=x, y=y)

    def prepare(self, tickers: List[str], save_dir: str, yahoo: bool = False, csv_path: str = './csvs/'):
        if yahoo:
            datasets = [self.from_yahoo(tick, save_dir) for tick in tickers]
        else:
            datasets = [self.read_csv(os.path.join(csv_path, f"{tick}.csv"), save_dir) for tick in tickers if os.path.isfile(os.path.join(csv_path, f"{tick}.csv"))]
        
        return [d.config for d in datasets if d.save()]