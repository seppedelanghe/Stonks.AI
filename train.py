import os, wandb, torch
import sys
import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader

from lib.model import ConvLSTM
from lib.loss import ConvLSTMLoss
from lib.data import StonksDataset
from lib.plots import make_compare_candle_plots, plot_terminal_graph
from lib.utils import load_checkpoint, save_checkpoint

TIME_D = 14
INPUT_LAYERS = 5
OUTPUT_LAYERS = 1

BATCH_SIZE = 512 * 4
SHUFFLE = False
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda"
AS_DOUBLE = False
DATA_WORKERS = 1

WAB = False
N_RENDER_IMAGES = 2

LOAD_MODEL = False
LOAD_MODEL_PATH = os.path.join('models', '07-23-2022', '3_checkpoint_00-40.pth')

TRAIN_CONFIG = f"./data/{TIME_D}_{OUTPUT_LAYERS}_config.json"
TEST_CONFIG = f"./data/{TIME_D}_{OUTPUT_LAYERS}_config.json"

m = ConvLSTM(TIME_D, INPUT_LAYERS, OUTPUT_LAYERS).to(DEVICE)
# m = ConvModel(TIME_D, INPUT_LAYERS, OUTPUT_LAYERS).to(DEVICE)
if AS_DOUBLE:
    m = m.double()

loss_fn = ConvLSTMLoss()
opt = Adam(m.parameters(), lr=LR)

if LOAD_MODEL:
    load_checkpoint(LOAD_MODEL_PATH, m, opt, DEVICE)


def categorize(x, y):
    col = 0
    return torch.round(y[:, col] - x[:, -1, col])

def test_fn(dataset: StonksDataset, items: int = 5, epoch: int = 1):
    loader = DataLoader(dataset, 4 * items, shuffle=True, num_workers=1)

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pred = m(x)

        x, y, y_pred = x.to('cpu').detach(), y.to('cpu').detach(), y_pred.to('cpu').detach()

        for i in range(items):

            # inverse scale before plotting
            curr_x = dataset.inverse_scale(x[i], with_vol=True)
            curr_y = dataset.inverse_scale(y[i], with_vol=False)
            curr_y_pred = dataset.inverse_scale(y_pred[i], with_vol=False)

            candles_save_path = make_compare_candle_plots(curr_x, curr_y, curr_y_pred, f"{epoch}_comparision_{i}.png", output_cols=OUTPUT_LAYERS)
            # gradient_save_path = make_color_gradient_compare_plot(x[i], y[i], y_pred[i], f("{ticker}_{i}_gradient.png", output_cols=OUTPUT_LAYERS)
            if WAB:
                wandb.log({
                    f"epoch_{epoch}_comparision_{i}": wandb.Image(candles_save_path),
                    # f"{ticker}_{i}_gradient": wandb.Image(gradient_save_path)
                })
            
        break
        
def train_fn(loader: DataLoader):
    losses, faults = [], 0

    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for i, (x, y) in loop:
        try:
            if len(losses) > 30 and i % 30 == 0:
                loop.set_postfix(loss=np.mean(losses[30:]), faults=faults)

            y = categorize(x, y)
            print(y)

            x, y = x.to(DEVICE), y.to(DEVICE)
            out = m(x)

            y = loader.dataset.inverse_scale(y, with_vol=False)
            out = loader.dataset.inverse_scale(out, with_vol=False)
            
            if torch.isnan(out).any():
                faults += 1
                continue

            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

            
        except KeyboardInterrupt as ke:
            raise ke
        except Exception as e:
            faults += 1

            raise e
        

    return np.mean(losses), faults

def test(epoch: int = 1):
    test_data = StonksDataset()
    test_data.load_datasets(TEST_CONFIG)
    test_fn(test_data, N_RENDER_IMAGES, epoch)


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

    train_data = StonksDataset()
    train_data.load_datasets(TRAIN_CONFIG)
    train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=SHUFFLE, num_workers=DATA_WORKERS)

    big_loop = tqdm(range(EPOCHS), leave=True)
    loss_over_time, fails = [], 0

    for epoch in big_loop:
        # test(epoch) # run before to check for faulty code before training

        try:
            mean_loss, faults = train_fn(train_loader)
            fails += faults

            big_loop.set_postfix({
                'fails': fails,
                'loss': mean_loss,
            })

            if WAB:
                wandb.log({
                    'loss': mean_loss,
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
    train()