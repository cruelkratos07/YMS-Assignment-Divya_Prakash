# Predicting a 2% move before the NSE open

A stock that moved 1.99% provides different training information from one that moved 0.2%, even though both receive the same binary label. The model uses that distinction. Ridge regression predicts the logarithm of the largest excursion from the previous close, measured in percentage points. At prediction time, the distance from the 2% boundary is scaled by the training residual standard deviation and blended with a fixed volatility and earnings score. The fitted component has weight 0.65. The decision threshold of 0.40 applies to the blended score.

## Features and sources

- **Stock volatility.** Inputs include implied volatility, its ratio to its one-month average, the stock's volatility relative to peers, the previous range and turnover relative to its 20-day median. A 31-calendar-day estimate uses Parkinson's daily variance, `log(high / low)² / (4 log 2)`, averaged over available prior ranges. Each as-of date counts once. This adapts the different horizons in Corsi's volatility model to a panel with missing sessions. Kambouroudis and colleagues also studied implied volatility and overnight returns in this setting. [Parkinson (1980)](https://www.cmegroup.com/trading/fx/files/michael_parkinson.pdf) · [Corsi (2009)](https://doi.org/10.1093/jjfinec/nbp001) · [Kambouroudis et al. (2021)](https://doi.org/10.1002/fut.22241)
- **Overnight price discovery.** India equity funds trade in Tokyo and Hong Kong before the NSE opens. Each available fund price is converted to rupees and scaled to the Nifty using up to three recent sessions when both markets traded. The median implied change from the previous Nifty close supplies a market-wide gap estimate. For five dual-listed stocks, a separate feature measures the return of their US depositary receipt while India's cash market was closed. Fund tracking differences and sparse quotes introduce uncertainty into the gap estimate. [Ou (2023)](https://doi.org/10.1080/10293523.2023.2208904) · [Tokyo Nifty 50 ETF](https://nextfunds.jp/en/lineup/1678/) · [Hong Kong SENSEX ETF](https://www.blackrock.com/hk/en/products/251590/)
- **Information cutoff.** The other market inputs include India VIX, US volatility, stock-index futures, Brent and Asian indexes. Hourly bars enter only after their full hour has ended, with a cutoff of 09:05 IST. Daily bars use earlier exchange dates. NSE earnings flags use the exchange dissemination time for released results and the notice timestamp for scheduled meetings, both strictly before 09:15 IST. Missing regression inputs take their training median. [Yahoo Finance](https://finance.yahoo.com/markets/) · [NSE announcements](https://www.nseindia.com/companies-listing/corporate-filings-announcements) · [NSE board meetings](https://www.nseindia.com/companies-listing/corporate-filings-board-meetings)

## Model selection

The comparison used expanding windows of whole training dates. After an eight-date warm-up, coefficients were refitted every two training dates using earlier sessions with usable implied volatility. Validation used the same 61 eligible dates for each model. The richer target and additional features improved the precision-weighted score while lowering precision. Among decision thresholds of 0.25, 0.40 and 0.55, the selected threshold gave the highest score for the revised model with at least 75% precision overall and over 60% in June–October.

| Training-date validation | Rows | Precision | Recall | Score |
|---|---:|---:|---:|---:|
| Previous binary model | 3,050 | 80.43% | 29.91% | 0.7520 |
| Excursion model | 3,050 | 75.86% | 37.59% | 0.7776 |
| Excursion model, June–October only | 2,050 | 61.96% | 8.91% | 0.4989 |

These rows were used for model selection. The June–October subset shows how much recall fell in calmer sessions. The earlier submission had already been graded on the official test set. The revision used the released training labels for all fits and comparisons, and its model and threshold were frozen before the repeat test run.

## Reproduction

```sh
python -m pip install -r requirements.txt
python prepare_user_historical_info.py
python fit_model.py
python evaluator/run_evaluation.py .
python evaluator/run_evaluation.py . --data evaluator/data
```

Prediction runs offline with the committed feature CSV and model. For each date it selects the last snapshot fitted strictly before that date. During warm-up it uses the fixed score. Rebuilding the model requires Python 3.12 or later. Rebuilding the feature CSV requires access to the data providers and their historical hourly bars.
