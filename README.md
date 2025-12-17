# volatility-forecasting-nbk

Given a time series of asset returns and potentially macro-economic markers, predict the volatility at a chosen time horizon. Then, we'll evaluate the out-of-sample performance of the DL model to see if it provides any material advantage over classic econometric methods.

## Runnable Scripts/Commands

### Training Models

**Train LSTM model:**
```bash
python scripts/train.py --model lstm
```
This trains the LSTM model and saves the best model weights to `models/lstm_best.pt`.

**Train GRU model:**
```bash
python scripts/train.py --model gru
```

**Run GARCH baseline:**
```bash
python scripts/run_baseline.py
```
This runs the GARCH(1,1) baseline model and prints the test RMSE.

### Evaluating and Comparing Models

**Compare models with 2020 cutoff:**
```bash
python scripts/compare_models.py --method 2020_cutoff
```
This compares GARCH and LSTM v2 models using a 2020-01-01 train/test split and generates comparison plots.

**Compare models with random out-of-sample split:**
```bash
python scripts/compare_models.py --method random_oos
```
This compares models using a random out-of-sample split.

**Backtest models:**
```bash
python -c "from src.evaluation import backtest_models; backtest_models('data/SPY_returns.csv', 'models/lstm_v2_best.pt')"
```
This runs backtesting on the trained models.

### Paper Trading Experiment

**Run paper trading experiment:**
```bash
python scripts/experiment.py
```
This uses the trained LSTM model to predict volatility and generates trading signals for VIXY (long volatility) or SVIX (short volatility) based on whether predicted volatility is higher or lower than current realized volatility.

**Options:**
- `--model-path`: Path to trained model file (default: `models/lstm_v2_best.pt`)
- `--trade-amount`: Dollar amount to trade (default: 1000.0)
- `--long-vol-etf`: Ticker for long volatility ETF (default: VIXY)
- `--short-vol-etf`: Ticker for short volatility ETF (default: SVIX)
- `--use-csv`: Use CSV file instead of live data from yfinance

**Example:**
```bash
python scripts/experiment.py --trade-amount 2000 --long-vol-etf VIXY --short-vol-etf SVIX
```

## Contribution

### Nathan Banketa
- Responsible for backtesting, comparing models, analyzing results, and creating paper trading experiment
- Files: `src/evaluation.py`, `scripts/compare_models.py`,`scripts/experiment.py`

### Nick Babukhadia
- Responsible for creating the models, training them, and analyzing results
- Files: `src/models/lstm.py`, `src/models/gru.py`, `scripts/train.py`

### Kwabena Osei-Bonsu
- Responsible for getting the data, creating the pipeline, and running the baseline GARCH model, and analyzing results from GARCH and our models
- Files: `src/data.py`, `src/models/garch.py`, `scripts/run_baseline.py`, `notebooks/baseline_garch.ipynb`
