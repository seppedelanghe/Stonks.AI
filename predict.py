import argparse


from typing import List
from torch.optim import Adam

from lib.model import LSTMModel
from lib.loss import LSTMLoss
from lib.data import StockDatasetConfig, StockDataset, StonksData
from lib.utils import load_checkpoint

parser = argparse.ArgumentParser(description="Train and use custom AI models for stock market predictions.")

parser.add_argument('--tickers', required=True, nargs="+", help="the ticker of the stock you want to predict the price for", type=str)
parser.add_argument('--inputs', nargs="+", help="the inputs to use in the model. The options are: Open, Close, Low, High. Default is Close.", type=str, default=['Close'])
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
COLUMNS = args.inputs
DEVICE = 'cpu'

'''
    Functions
'''

def predict(configs: List[StockDatasetConfig], columns: List[str]):
    '''
        Setup
    '''

    m = LSTMModel(len(columns), TIME, LAYERS, len(columns), DEVICE).to(DEVICE)
    opt = Adam(m.parameters(), lr=0)

    try:
        load_checkpoint(args.model, m, opt, DEVICE)
    except Exception as e:
        raise Exception('the params TIME and/or LAYERS are not correct for this model. Put in the correct values for this model or train a model with these parameters.')

    results = {}

    for config in configs:
        dataset = StockDataset.load(config.config_path)
        
        x = dataset.X[-1:]
        y = m(x)

        curr = [float(val) for val in StonksData.inverse_normalize(x[-1, -1].squeeze(), config.ranges[0], config.ranges[1])]
        pred = [float(val) for val in StonksData.inverse_normalize(y.detach(), config.ranges[0],config.ranges[1]).squeeze()]
        results[config.ticker] = [
            curr,
            pred,
            columns
        ]

    return results

def print_results(results: dict):
    print("".join('-' for _ in range(64)))

    for ticker, (curr, pred, cols) in results.items():
        print(f"{ticker}:")
        print(f"\tToday:")
        for i, c in enumerate(curr):
            print(f"\t\t\t{cols[i]}:\t{c}")
        print(f"\tPredictions:")
        for i, p in enumerate(pred):
            print(f"\t\t\t{cols[i]}:\t{p}")
        print("".join('-' for _ in range(64)))

if __name__ == "__main__":
    sd = StonksData(TIME, COLUMNS, prediction_mode=True)
    configs = sd.prepare(TICKERS, './data', yahoo=True)
    results = predict(configs, sd.cols)
    print_results(results)
