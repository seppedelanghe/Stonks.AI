import os, torch, io
import plotly.graph_objects as go
from PIL import Image
import numpy as np

def make_candle_plot(x, y, name: str = 'prediction'):
    data = torch.vstack((x, y.reshape(1, 5)))

    pred_x = len(x) - 0.5
    fig = go.Figure(data=[go.Candlestick(x=[i for i in range(len(x) + 1)],
                open=data[:, 1],
                high=data[:, 2],
                low=data[:, 0],
                close=data[:, 3])]).update_layout(
                title=f"Stonks {name}",
                        xaxis_title="Time",
                        yaxis_title="Price, $",
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


def make_compare_candle_plots(x, y_true, y_pred, name: str = 'comparison.png'):
    path = os.path.join('results', name)

    g_true = make_candle_plot(x, y_true, 'actual')
    g_pred = make_candle_plot(x, y_pred, 'prediction')

    im_true = Image.open(io.BytesIO(g_true))
    im_pred = Image.open(io.BytesIO(g_pred))
    im_comb = Image.fromarray(np.vstack((np.array(im_true), np.array(im_pred))))
    
    im_comb.save(path)

    return path