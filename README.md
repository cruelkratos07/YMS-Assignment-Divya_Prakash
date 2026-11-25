# YMS Financials Assignment

Predict before the NSE opens which Nifty 50 stocks will trade more than 2% away from their previous close that day.

A stock qualifies if its intraday high rises more than 2% above its previous close or its intraday low falls more than 2% below it. This Python project produces one prediction per stock for a given date and includes an evaluator to measure how well those predictions identify qualifying moves.

<!-- TODO: Add a build badge when a CI workflow is available. -->
<!-- TODO: Add a version badge when a release is available. -->
<!-- TODO: Add a license badge when a license is provided. -->

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- Use the committed model and feature CSV to run predictions offline with the Python standard library.
- Estimate each stock’s largest move from its previous close with ridge regression. The model combines prior trading ranges, implied volatility, turnover, overnight market signals, and earnings announcements.
- Score predictions on the supplied training or test dates. The evaluator reports precision, recall, and a score that gives precision more weight.

See [NOTES.md](NOTES.md) for feature sources, model fitting, and threshold selection.

## Installation

Use Python 3.10 or later to run predictions and evaluation. Use Python 3.12 or later to rebuild the model. Cloning requires Git and access to the private GitHub repository.

1. Clone the repository and enter it.

   ```sh
   git clone https://github.com/cruelkratos07/YMS-Assignment-Divya_Prakash.git
   cd YMS-Assignment-Divya_Prakash
   ```

2. Install the dependencies if you plan to regenerate features or rebuild the model. You can go straight to the quick start to use the committed files.

   ```sh
   python -m pip install -r requirements.txt
   ```

   [requirements.txt](requirements.txt) lists `tzdata` for Windows and `numpy`, `scikit-learn`, and `threadpoolctl` for model fitting on Python 3.12 or later.

## Quick start

Run the predictor for a date included in the repository.

```sh
python predict.py --date 20260420 --historical-info data/historical_info --user-historical-info user_historical_info.csv --out-dir predictions
```

The command writes `predictions/20260420.predictions.csv` with 50 rows and two columns. `name` is the stock symbol. `prediction` is `True` when the model flags a move beyond 2% and `False` otherwise.

To evaluate the committed model across multiple dates, run either command from the repository root.

```sh
python evaluator/run_evaluation.py .
python evaluator/run_evaluation.py . --data evaluator/data
```

The first command evaluates the training dates. The second evaluates the supplied test dates. Both write per-date predictions to `eval_predictions/` and aggregate results to `eval_score.json`. The evaluator’s default pass condition is precision strictly above 60%.

## Configuration

### Prediction options

| Option | Required | Purpose |
| --- | --- | --- |
| `--date` | Yes | Trading date in `YYYYMMDD` format. |
| `--historical-info` | Yes | Directory containing `<YYYYMMDD>_n50.info.csv` files. |
| `--user-historical-info` | No | CSV containing additional pre-open features. |
| `--out-dir` | Yes | Directory for `<YYYYMMDD>.predictions.csv`. |

The predictor reads `model.json` from the same directory as `predict.py`.

### Feature generation and model fitting

| Script | Option | Default | Purpose |
| --- | --- | --- | --- |
| `prepare_user_historical_info.py` | `--dates` | `data/all_dates.txt` | File listing the trading dates. |
| `prepare_user_historical_info.py` | `--names` | `data/n50_names.txt` | File listing the stock symbols. |
| `prepare_user_historical_info.py` | `--out` | `user_historical_info.csv` | Destination for the feature CSV. |
| `fit_model.py` | `--data` | `data` | Directory containing the training labels and historical stock data. |
| `fit_model.py` | `--market-info` | `user_historical_info.csv` | Input CSV containing additional pre-open features. |
| `fit_model.py` | `--out` | `model.json` | Destination for the fitted model snapshots. |

<!-- TODO: Document environment variables if the project adds any. -->

## Project structure

```text
.
├── data/
│   ├── all_dates.txt                # Trading dates covered by the feature CSV
│   ├── historical_info/            # Daily pre-open stock data for training dates
│   ├── n50_names.txt                # Stock symbols
│   └── train_labels.csv             # Training labels and maximum excursions
├── evaluator/
│   ├── data/                        # Inputs and labels for the supplied test evaluation
│   └── run_evaluation.py            # Runs predictions by date and aggregates scores
├── fit_model.py                     # Fits dated ridge regression snapshots
├── model.json                       # Committed model snapshots and decision threshold
├── NOTES.md                         # Features, methodology, and validation
├── overnight.py                     # Overseas ETF and depositary receipt features
├── predict.py                       # Single-date prediction CLI
├── prepare_user_historical_info.py   # Builds the pre-open feature CSV
├── requirements.txt                 # Dependencies for regeneration and model fitting
├── test_predictions.py              # Scores prediction files against labeled outcomes
└── user_historical_info.csv         # Committed pre-open features
```

## Contributing

For local development, use Python 3.12 or later and install the dependencies. The training-date evaluator provides an end-to-end check.

```sh
python -m pip install -r requirements.txt
python evaluator/run_evaluation.py .
```

To regenerate the feature CSV and rebuild the model, run the commands below. They replace the committed CSV and model files in your working copy. Feature generation requires network access to Yahoo Finance and NSE, including access to historical hourly bars.

```sh
python prepare_user_historical_info.py
python fit_model.py
```

<!-- TODO: Add a contribution policy and unit-test instructions if the repository adopts them. -->

## License

<!-- TODO: Add the license name and link to LICENSE when a license file is provided. -->
