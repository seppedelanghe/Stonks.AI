from datetime import datetime
import os, wandb, torch
from random import randint
import sys
import numpy as np

from tqdm import tqdm
from torch.optim import Adam

from lib.model import ConvLSTM
from lib.loss import ConvLSTMLoss
from lib.data import get_all_csv_files, get_loader_for_file
from lib.plots import make_color_gradient_compare_plot, make_compare_candle_plots

DATA_PATH = './stock_market_data'
FILES = get_all_csv_files(DATA_PATH)
EXCEPTIONS = []

TIME_D = 10
INPUT_LAYERS = 6
OUTPUT_LAYERS = 5
BATCH_SIZE = 512
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda"
AS_DOUBLE = False
WAB = False

LOAD_MODEL = False
LOAD_MODEL_PATH = os.path.join('models', 'sp500_413_loss_688.tar.pth')

m = ConvLSTM(TIME_D, INPUT_LAYERS, OUTPUT_LAYERS).to(DEVICE)
if AS_DOUBLE:
    m = m.double()

if LOAD_MODEL:
    m.load_state_dict(torch.load(LOAD_MODEL_PATH))

loss_fn = ConvLSTMLoss()
opt = Adam(m.parameters(), lr=LR)


def train_fn(loader):
    losses = []

    for x, y in loader:            
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = m(x)
        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

    if WAB:
        wandb.log({"loss": int(np.mean(losses))})

    return int(np.mean(losses))

def test_fn(loader, items: int = 5, ticker: str = 'unknown'):
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pred = m(x)

        x, y, y_pred = x.to('cpu').detach(), y.to('cpu').detach(), y_pred.to('cpu').detach()
        for i in range(items):
            gradient_save_path = make_color_gradient_compare_plot(x[i], y[i], y_pred[i], f"{ticker}_{i}_gradient.png", output_cols=OUTPUT_LAYERS)
            candles_save_path = make_compare_candle_plots(x[i], y[i], y_pred[i], f"{ticker}_{i}_comparision.png", output_cols=OUTPUT_LAYERS)
            if WAB:
                wandb.log({
                    f"{ticker}_{i}_comparision": wandb.Image(candles_save_path),
                    f"{ticker}_{i}_gradient": wandb.Image(gradient_save_path)
                })
            
            if i == items:
                break

        break
        

def test(file_n: int = -1):
    print("=> Running test predicitons")

    if file_n == -1:
        file_n = randint(0, len(FILES))

    _, test_loader = get_loader_for_file(FILES[file_n], BATCH_SIZE, TIME_D, AS_DOUBLE)
    ticker = os.path.basename(FILES[file_n])[:-4]
    test_fn(test_loader, 5, ticker)

    print("=> Test prediction results saved")

def train():
    if WAB:
        wandb.init(project='Stonks', entity="seppedelanghe",
            config={
                "learning_rate": LR,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "depth": TIME_D,
                "double": AS_DOUBLE
        })

    big_loop = tqdm(range(EPOCHS), leave=True)
    for epoch in big_loop:
        test() # run before to check for faulty code before training
        
        fails = 0
        loop = tqdm(enumerate(FILES), total=len(FILES), leave=False)
        for i, path in loop:
            if path in EXCEPTIONS:
                continue

            try:
                train_loader, _ = get_loader_for_file(path, BATCH_SIZE, TIME_D, AS_DOUBLE)
                mean_loss = train_fn(train_loader)
                loop.set_postfix(loss=mean_loss) # update progress bar
                big_loop.set_postfix(fails=fails)
            except KeyboardInterrupt:
                print('Control-C pressed, stopping...')
                sys.exit()
            except Exception:
                fails += 1
                EXCEPTIONS.append(path)
                continue
        

    now = str(datetime.now())
    torch.save(m.state_dict(), os.path.join('models', f'sp500_413_{now}.tar.pth'))

if __name__ == '__main__':
    train()