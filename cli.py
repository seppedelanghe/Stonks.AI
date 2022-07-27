import argparse, sys

import numpy as np

from typing import List
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader

from lib.model import LSTMModel
from lib.loss import LSTMLoss
from lib.data import StockDatasetConfig, StockDataset, StonksData
from lib.plots import plot_terminal_graph
from lib.utils import load_checkpoint, save_checkpoint

parser = argparse.ArgumentParser(description="Train and use custom AI models for stock market predictions.")

parser.add_argument('--tickers', required=True, nargs="+", help="the tickers of the stocks the model should train for.", type=str)
parser.add_argument('--time', help='how many days to account for in the model. 30 is a good start point.', type=int, default=30)
parser.add_argument('--layers', help='amount of internal layers in the LSTM. Default is 2.', type=int, default=2)
parser.add_argument('--batch', help='the batch size for training. this depends how much ram you have, something between 4 and 16 is usually a good starting point.', type=int, default=8)
parser.add_argument('--epochs', help='for how many iterations should the model train', type=int, default=10)
parser.add_argument('--lr', help="the learning rate decides how fast the model should learn from it's findings. 0.001 is the default", type=float, default=1e-3)

parser.add_argument('--cuda', action=argparse.BooleanOptionalAction, help='use cuda to train the model. Only supported on systems with a Nvidia GPU.')

args = parser.parse_args()

'''
    Hyperparams
'''
TIME = args.time
LAYERS = args.layers
EPOCHS = args.epochs
BATCH_SIZE = args.batch
LR = args.lr
DEVICE = "cuda" if args.cuda else 'cpu'
TICKERS = args.tickers

SHUFFLE = True
DATA_WORKERS = 1
CSV_PATH = 'stock_market_data/forbes2000/csv/'


'''
    Setup
'''

m = LSTMModel(1, TIME, LAYERS, 1, DEVICE).to(DEVICE)


loss_fn = LSTMLoss()
opt = Adam(m.parameters(), lr=LR)

stonks = StonksData(TIME)
configs = stonks.prepare(TICKERS, './data/', csv_path=CSV_PATH)

'''
    Functions
'''

def train_fn(configs: List[StockDatasetConfig]):
    losses = []

    for config in configs:
        try:
            dataset = StockDataset.load(config.config_path)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=DATA_WORKERS, pin_memory=True)

            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = m(x)

                loss: torch.Tensor = loss_fn(out, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

                losses.append(loss.item())

        except KeyboardInterrupt as ke:
            raise ke
        except Exception as e:
            raise e

    return np.mean(losses)

def train():
    big_loop = tqdm(range(EPOCHS), leave=True)
    loss_over_time = []

    for epoch in big_loop:
        try:
            mean_loss = train_fn(configs)
            big_loop.set_postfix({'loss': mean_loss})

            loss_over_time.append(mean_loss)
            save_checkpoint(m, opt, epoch, "_".join(TICKERS))

        except KeyboardInterrupt:
            print('Control-C pressed, stopping...')
            sys.exit()
        except Exception as e:
            raise e

    plot_terminal_graph(np.array(loss_over_time), 'loss')


if __name__ == "__main__":
    print(f"=>\tStarting model training...")
    train()
    print(f"=>\tTraining finished!")
