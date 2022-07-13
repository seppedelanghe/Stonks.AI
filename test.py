import os
import numpy as np
import pandas as pd

import plotly as py
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px

from datetime import timedelta, datetime

from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

FORBES = './stock_market_data/sp500/csv/'
TICKER = 'PFE'
PATH = os.path.join(FORBES, f"{TICKER}.csv")

df = pd.read_csv(PATH)
df['Date'] = pd.to_datetime(df['Date'])

start_date = df.Date.iloc[-1] - timedelta(days=7)
df = df[df.Date > start_date]

def make_dataset(df: pd.DataFrame, its: int = 6):
    X, y = [], []
    last = []
    for i, row in df.iterrows():
        if type(last) == type(None) or len(last) < 6:
            last.append(row.values)
            continue
        X.append(np.array(last))
        y.append(row.values)
        last.pop(0)
        last.append(row.values)

    return np.array(X), np.array(y)

X, y = make_dataset(df.drop(['Date'], axis=1))

from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, InputLayer, Flatten, Reshape
from tensorflow.keras.models import Sequential

BATCH_SIZE = 2

m = Sequential()

m.add(InputLayer(input_shape=(6, 6)))

m.add(LSTM(6, activation='tanh', dropout=0.1))
m.add(BatchNormalization())

m.add(Dense(6 * 6, activation='relu'))
m.add(BatchNormalization())

m.add(Dense(6, activation='softmax'))


m.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
m.fit(X, y, batch_size=16, epochs=5)