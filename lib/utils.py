import os, torch
from datetime import datetime
import pandas as pd
from tqdm import tqdm

def prod(val):
    res = 1 
    for ele in val: 
        res *= ele 
    return res

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unicode_block(s: int):
    assert s > 0 and s < 9, 's to small or too large, needs to be 1-8'
    return [
        u'\u2581',
        u'\u2582',
        u'\u2583',
        u'\u2584',
        u'\u2585',
        u'\u2586',
        u'\u2587',
        u'\u2588',
    ][s-1]

def save_checkpoint(model, optim, epoch: int = 1, ticker: str = None):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optim': optim.state_dict()
    }
    today = datetime.now().strftime("%m-%d-%Y")
    now = datetime.now().strftime("%H-%M")
    os.makedirs(os.path.join('models', today), exist_ok=True)
    torch.save(checkpoint, os.path.join('models', today, f'{ticker}_{epoch}_checkpoint.pth' if type(ticker) != type(None) else f'{epoch}_checkpoint_{now}.pth'))

def load_checkpoint(path: str, model, optim, device: str = 'cuda'):
    assert os.path.isfile(path), 'Checkpoint file does not exist.'
    checkpoint = torch.load(path, map_location=torch.device(device))
    model.load_state_dict(checkpoint.get('model'))
    optim.load_state_dict(checkpoint.get('optim'))

    ''''
        temp fix for bug in pytorch => should be fixed in next minor update
        https://github.com/pytorch/pytorch/issues/80809
    '''
    optim.param_groups[0]['capturable'] = True

def calc_diff(y, y_pred, precision: int = 4):
    precision = 10 ** precision
    return float(torch.mean(input=(torch.round(y * precision) / precision) - (torch.round(y_pred * precision) / precision))) * 100

def get_all_csv_files(dir: str):
    files = []
    a = os.listdir(dir)
    for b in a:
        if os.path.isdir(os.path.join(dir, b)):
            c = os.path.join(dir, b, 'csv')
            if os.path.isdir(c):
                files += [os.path.join(c, f) for f in os.listdir(c) if f[-4:] == '.csv']

    return files

def large_df(path):
    files = get_all_csv_files(path)
    return pd.concat((pd.read_csv(f) for i, f in tqdm(enumerate(files), total=len(files), leave=False)), ignore_index=True)