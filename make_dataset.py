import argparse
import sys
from lib.data import StonksDataset
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('time_d', type=int)
parser.add_argument('outputs', type=int, default=4)
parser.add_argument('workdir', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    stonks = StonksDataset(time_d=args.time_d, output_params=args.outputs)
    data = stonks.from_csvs(args.workdir)
    stonks.make_datasets(data)
    stonks.save_datasets('./data/')
    
    
