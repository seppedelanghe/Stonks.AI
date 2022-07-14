import os, torch, io
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from datetime import datetime

RESULTS_PATH = os.path.join('results', datetime.now().strftime("%m-%d-%Y"))
os.makedirs(RESULTS_PATH, exist_ok=True)

def make_candle_plot(x, y, name: str = 'prediction', output_cols: int = 5):
    data = torch.vstack((x[:, :output_cols], y.reshape(1, output_cols)))

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


def make_color_gradient_plot(x, title: str, lbls = ['Low', 'Open', 'High', 'Close', 'Adjusted Close'], cmap: str = 'autumn'):
    assert x.shape[1] == len(lbls), 'Labels need to be the same length as the type of values in x'

    f, ax = plt.subplots(x.shape[1], 1, figsize=(11, 6), dpi=150)
    f.suptitle(title, fontsize=16)
    for i in range(x.shape[1]):
        ax[i].imshow(x[:, 0].reshape(1, x.shape[0]), cmap=cmap, vmin=0, vmax=255)
        ax[i].set_ylabel(lbls[i], rotation=0, labelpad=40)
        ax[i].xaxis.set_visible(False)
        ax[i].yaxis.set_ticks([])

    return f

def make_color_gradient_compare_plot(x, y_true, y_pred, name: str = "color_gradient.png", output_cols: int = 5):
    path = os.path.join(RESULTS_PATH, name)

    x_true = torch.vstack((x[:, :output_cols], y_true)) * 255 # times 255 for colors

    g_true = make_color_gradient_plot(x_true, 'actual')
    w, h = g_true.get_size_inches() * g_true.get_dpi()
    w, h = int(w), int(h)
    canvas = FigureCanvas(g_true)
    canvas.draw()
    im_true = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
    
    x_pred = torch.vstack((x[:, :output_cols], y_pred)) * 255 # times 255 for colors
    g_pred = make_color_gradient_plot(x_pred, 'prediction')
    w, h = g_pred.get_size_inches() * g_pred.get_dpi()
    w, h = int(w), int(h)
    canvas = FigureCanvas(g_pred)
    canvas.draw()
    im_pred = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)

    im_comb = Image.fromarray(np.vstack((im_true, im_pred)))
    
    im_comb.save(path)

    return path