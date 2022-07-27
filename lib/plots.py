import os, torch, io
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from datetime import datetime

from lib.utils import unicode_block

RESULTS_PATH = os.path.join('results', datetime.now().strftime("%m-%d-%Y"))
os.makedirs(RESULTS_PATH, exist_ok=True)

def plot_terminal_graph(x, name: str, max_size: int = 80):
    if type(x) == torch.Tensor:
        x = x.to_numpy()
    
    h = np.max(x)
    idxs = np.rint(np.linspace(0, x.shape[0] - 1, max_size)).astype('int16')
    line = ''.join([unicode_block(int(x[idx]/h*7)+1) for idx in idxs])
    print(f"{name}:\t", line)

def make_candle_plot(x: torch.Tensor, y: torch.Tensor, name: str = 'prediction', output_cols: int = 5):
    inp = x[:, :-1]
    out = y.reshape(1, output_cols)

    data = torch.vstack((inp, out))

    pred_x = len(x) - 0.5
    fig = go.Figure(data=[go.Candlestick(x=[i for i in range(len(x) + 1)],
                open=data[:, 1],
                high=data[:, 2],
                low=data[:, 0],
                close=data[:, 3])]).update_layout(
                title=f"Stonks {name}",
                        xaxis_title="Time",
                        yaxis_title="Price normalized",
                        font=dict(
                            family="Courier New, monospace",
                            size=18,
                            color="RebeccaPurple"
                        ),
                        shapes = [dict(
                            x0=pred_x, x1=pred_x + 1, y0=0, y1=1, xref='x', yref='paper',
                            line_width=2)],
                        annotations=[
                            dict(
                                x=pred_x,
                                y=1,
                                xref="x", yref="paper",
                                showarrow=True, xanchor='left', text=name
                            )
                        ]
                    )        

    return fig.to_image(width=720, height=480)


def make_compare_candle_plots(x, y_true, y_pred, name: str = 'comparison.png', output_cols: int = 5):
    path = os.path.join(RESULTS_PATH, name)

    g_true = make_candle_plot(x, y_true, 'actual', output_cols=output_cols)
    g_pred = make_candle_plot(x, y_pred, 'prediction', output_cols=output_cols)

    im_true = Image.open(io.BytesIO(g_true))
    im_pred = Image.open(io.BytesIO(g_pred))
    im_comb = Image.fromarray(np.vstack((np.array(im_true), np.array(im_pred))))
    
    im_comb.save(path)

    return path

def make_stock_price_timeline(y_true, y_pred, ticker: str, savepath: str = 'price_timeline.png'):
    # Visualising the results
    figure, axes = plt.subplots(figsize=(15, 6))

    axes.plot(y_true[:, 0], color = 'red', label = f"Real {ticker} Stock Price")
    axes.plot(y_pred[:, 0], color = 'blue', label = f"Predicted {ticker} Stock Price")

    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel('Time')
    plt.ylabel(f"{ticker} Stock Price")
    plt.legend()
    plt.savefig(savepath)
