import argparse


from typing import List
from torch.optim import Adam

from lib.model import LSTMModel
from lib.loss import LSTMLoss
from lib.data import StockDatasetConfig, StockDataset, StonksData
from lib.utils import load_checkpoint

parser = argparse.ArgumentParser(description="Train and use custom AI models for stock market predictions.")

parser.add_argument('--tickers', required=True, nargs="+", help="the ticker of the stock you want to predict the price for", type=str)
parser.add_argument('--model', required=True, help='path to the saved model checkpoint to load', type=str)
parser.add_argument('--time', help='days before today you want to pass through the model', type=int, default=30)
parser.add_argument('--layers', help='amount of internal layers in the LSTM. Default is 2.', type=int, default=2)

args = parser.parse_args()

'''
    Hyperparams
'''
TIME = args.time
LAYERS = args.layers
TICKERS = args.tickers
DEVICE = 'cpu'


'''
    Setup
'''

m = LSTMModel(1, TIME, LAYERS, 1, DEVICE).to(DEVICE)

loss_fn = LSTMLoss()
opt = Adam(m.parameters(), lr=0)


'''
    Functions
'''

def collect_data():
    sd = StonksData(TIME, prediction_mode=True)
    return sd.prepare(TICKERS, './data', yahoo=True)

def predict(configs: List[StockDatasetConfig]):
    results = {}

    for config in configs:
        dataset = StockDataset.load(config.config_path)
        
        x = dataset.X[-1:]
        y = m(x)

        results[config.ticker] = [
            float(StonksData.inverse_normalize(x[-1, -1].squeeze(), config.ranges[0], config.ranges[1])),
            float(StonksData.inverse_normalize(y.detach(), config.ranges[0],config.ranges[1]).squeeze())
        ]

    return results

def print_results(results: dict):
    for ticker, val in results.items():
        print(f"{ticker}:\ncurrent:\t{val[0]}\nprediction:\t{val[1]}")

if __name__ == "__main__":
    
    try:
        load_checkpoint(args.model, m, opt, DEVICE)
    except Exception as e:
        raise Exception('the params TIME and/or LAYERS are not correct for this model. Put in the correct values for this model or train a model with these parameters.')

    configs = collect_data()
    results = predict(configs)
    print_results(results)
