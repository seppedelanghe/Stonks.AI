import os, wandb, torch
import sys
import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from random import randint

from lib.model import ConvLSTM
from lib.loss import ConvLSTMLoss
from lib.data import PreprocessedDataset
from lib.plots import make_compare_candle_plots, plot_terminal_graph
from lib.utils import load_checkpoint, save_checkpoint, calc_diff

TIME_D = 10
INPUT_LAYERS = 5
OUTPUT_LAYERS = 4

BATCH_SIZE = 512
SHUFFLE = True
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda"
AS_DOUBLE = False
DATA_WORKERS = 1

WAB = False
N_RENDER_IMAGES = 2

LOAD_MODEL = False
LOAD_MODEL_PATH = os.path.join('models', 'checkpoint_epoch_9_07-15-2022_21-17.tar.pth')

m = ConvLSTM(TIME_D, INPUT_LAYERS, OUTPUT_LAYERS).to(DEVICE)
if AS_DOUBLE:
    m = m.double()

loss_fn = ConvLSTMLoss()
opt = Adam(m.parameters(), lr=LR)

if LOAD_MODEL:
    load_checkpoint(LOAD_MODEL_PATH, m, opt, DEVICE)


train_data = PreprocessedDataset('./data/X.bin', './data/y.bin', TIME_D, OUTPUT_LAYERS)
train_data.set_scale(0.0, 44500.0, 0.0, 7421640800.0)
train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=SHUFFLE, num_workers=DATA_WORKERS)

def test_fn(loader, items: int = 5):
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pred = m(x)

        x, y, y_pred = x.to('cpu').detach(), y.to('cpu').detach(), y_pred.to('cpu').detach()
        for i in range(items):
            candles_save_path = make_compare_candle_plots(x[i], y[i], y_pred[i], f"{i}_comparision.png", output_cols=OUTPUT_LAYERS)
            # gradient_save_path = make_color_gradient_compare_plot(x[i], y[i], y_pred[i], f"{ticker}_{i}_gradient.png", output_cols=OUTPUT_LAYERS)
            if WAB:
                wandb.log({
                    f"{i}_comparision": wandb.Image(candles_save_path),
                    # f"{ticker}_{i}_gradient": wandb.Image(gradient_save_path)
                })
            
            if i == items:
                break
        break
        
def train_fn(loader):
    losses, acc, faults = [], [], 0

    loop = tqdm(loader, leave=False)
    for x, y in loop:
        try:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = m(x)

            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            acc.append(calc_diff(y, out))
            losses.append(loss.item())
            loop.set_postfix(loss=loss.item()) # update progress bar
        except KeyboardInterrupt as ke:
            raise ke
        except Exception as e:
            faults += 1
            continue
        

    return np.mean(losses), np.mean(acc), faults

def test():
    test_fn(train_loader, N_RENDER_IMAGES)

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
    loss_over_time, fails = [], 0

    for epoch in big_loop:
        # test() # run before to check for faulty code before training

        try:
            mean_loss, mean_diff, faults = train_fn(train_loader)
            fails += faults

            big_loop.set_postfix({
                'fails': fails,
            })

            if WAB:
                wandb.log({
                    'loss': mean_loss,
                    'diff': mean_diff,
                    'fails': fails
                })
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
    # print(count_parameters(m))
    # print(m.mid_neurons)

    train()