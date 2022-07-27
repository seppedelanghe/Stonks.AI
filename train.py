import os
import torch
import wandb
from random import randint
from typing import List
import sys
import numpy as np
from datetime import datetime

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from lib.loss import LSTMLoss

from lib.model import LSTMModel
from lib.data import StockDatasetConfig, StockDataset, StonksData
from lib.plots import make_stock_price_timeline, plot_terminal_graph
from lib.utils import save_checkpoint


'''
    Hyperparams
'''
TIME = 30
LAYERS = 2
EPOCHS = 10
BATCH_SIZE = 8
SHUFFLE = True
LR = 1e-4
DEVICE = "cuda"
DATA_WORKERS = 1

WAB = False

CSV_PATH = 'stock_market_data/forbes2000/csv/'
TICKERS = ['AAPL', 'CAT', 'TSLA']


'''
    Setup
'''

m = LSTMModel(1, TIME, LAYERS, 1, DEVICE).to(DEVICE)

loss_fn = LSTMLoss()
opt = Adam(m.parameters(), lr=LR)

stonks = StonksData(TIME)
configs = stonks.prepare(TICKERS, './data/', csv_path=CSV_PATH)


def test_fn(dataset: StockDataset, epoch: int = 1):
    x, y = dataset.X.to(DEVICE), dataset.y.to(DEVICE)
    y_pred: torch.Tensor = m(x)

    y, y_pred = y.to('cpu').detach(), y_pred.to('cpu').detach()

    filename = f"{epoch}_comparision_{dataset.config.ticker}.png"
    today = datetime.now().strftime("%m-%d-%Y")
    save_path = os.path.join('results', today)
    os.makedirs(save_path, exist_ok=True)
    make_stock_price_timeline(y, y_pred, dataset.config.ticker, os.path.join(save_path, filename))

    if WAB:
        wandb.log({
            f"{epoch}_comparision_{dataset.config.ticker}": wandb.Image(filename)
        })


def train_fn(configs: List[StockDatasetConfig]):
    losses = []

    loop = tqdm(configs, total=len(configs), leave=False)
    for config in loop:
        try:
            dataset = StockDataset.load(config.config_path)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=DATA_WORKERS, pin_memory=True)

            for x, y in loader:

                x, y = x.to(DEVICE), y.to(DEVICE)
                out = m(x)

                loss = loss_fn(out, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

                losses.append(loss.item())
            
            
            loop.set_postfix(loss=np.mean(losses))


        except KeyboardInterrupt as ke:
            raise ke
        except Exception as e:
            raise e

    return np.mean(losses)

def test(epoch: int = 1):
    dataset = StockDataset.load(configs[randint(0, len(configs) - 1)].config_path)
    test_fn(dataset, epoch)

def train():
    if WAB:
        wandb.init(project='Stonks', entity="seppedelanghe",
            config={
                "learning_rate": LR,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "time depth": TIME,
                "layers": LAYERS
        })

    big_loop = tqdm(range(EPOCHS), leave=True)
    loss_over_time = []

    for epoch in big_loop:
        try:
            test(epoch) # run before to check for faulty code before training
            
            mean_loss = train_fn(configs)
            big_loop.set_postfix({'loss': mean_loss})

            if WAB:
                wandb.log({'loss': mean_loss})
            else:
                loss_over_time.append(mean_loss)

            save_checkpoint(m, opt, epoch)

        except KeyboardInterrupt:
            print('Control-C pressed, stopping...')
            sys.exit()
        except Exception as e:
            raise e

    if not WAB:
        plot_terminal_graph(np.array(loss_over_time), 'loss')

if __name__ == '__main__':
    train()