# Stonks AI

![Quality badge](https://img.shields.io/badge/quality-decent-green)

Attempt to predict stock prices using PyTorch with command line support. <br>
The results from the model are by far not reliable, __just a fun project.__

# Example

Prediction from 28/07/2022 for multiple NASDAQ stocks:

[![stock-predictions.png](https://i.postimg.cc/WbWp1sbd/stock-predictions.png)](https://postimg.cc/ppjM0bjv)

# Requirements

- Python 3.9
- PyTorch 1.12.0
    - CPU version
    - or
    - CUDA 11.3 version (GPU training only)
- pip packages in `requirements.txt`
- stock data in csv format

__predicting only__
- a pretrained model

<br>

# Training

Train a model on Apple stock for 20 epochs with CUDA acceleration: \
`python cli.py --tickers AAPL --epochs 20 --lr 0.0001 --cuda`
<br><br>
Train a model on Microsoft and Google stocks for 30 epochs and a depth of 3 on CPU: \
`python cli.py --tickers MSFT GOOG --epochs 30 --layers 3 --lr 0.0001`
<br><br>
To view all parametes for training: \
`python cli.py --help`

<br><br>
# Predicting

Predicting gets the latest stock data from Yahoo finance, no csv files needed.

Predicting Apples stock price for the next day using a pretrained model: \
`python predict.py --tickers AAPL --model '.\models\AAPL_model.pth'`
<br><br>
To view all parametes for predicting: \
`python predict.py --help`
<br><br>

# Roadmap

- [x] Multi dim predictions => close, open, low and high
- [ ] Improve loss function
- [ ] Add candle plots back
- [ ] Add Weights and Biases support