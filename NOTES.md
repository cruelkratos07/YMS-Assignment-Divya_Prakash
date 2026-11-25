# Nifty-50 opening risk model

The model flags stocks likely to trade more than 2% away from their previous close in either direction. It blends a ridge-regularized logistic regression with a volatility and earnings risk score. The regression uses absolute and peer-relative stock implied volatility, the prior session's range, turnover relative to its 20-day median, and a regularized offset for each symbol. The blend places 65% weight on the fitted logit and flags a score of 0.55 or higher.

## Market and earnings inputs

- The feature CSV uses completed daily bars for India VIX, the S&P 500, Cboe VIX and the US-listed iShares India 50 ETF. The previous Nifty 50 session supplies the anchor for overnight returns and the earnings window. [Yahoo Finance market data](https://finance.yahoo.com/markets/), [iShares INDY](https://www.ishares.com/us/products/239758/ishares-india-50-etf)
- E-mini S&P 500 and Brent returns run from the previous NSE close to the latest usable pre-open bar. Nikkei and Hang Seng returns run from the previous Asian session to the pre-open bar. An hourly quote enters the CSV only if its full one-hour window ends by 09:05 IST. An absent quote stays blank. [CME E-mini S&P 500 futures](https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.html)
- The earnings-release flag covers financial results disseminated by NSE between the previous NSE close and 09:15 IST. A scheduled results meeting is eligible when its exchange notice was public before the prediction cutoff. [NSE announcements](https://www.nseindia.com/companies-listing/corporate-filings-announcements), [NSE board meeting notices](https://www.nseindia.com/companies-listing/corporate-filings-board-meetings)

## Validation and timing

I compared stock-only, market and overnight feature sets using expanding windows of whole training dates. Each fit uses earlier training dates with usable stock volatility. The model refits every two training dates after an eight-date warm-up. I chose the threshold for the grader's precision-weighted score and checked the quieter June–October training period. The selected setup reached **80.43% precision, 29.91% recall and 0.7520 score** across 3,050 expanding-window validation rows. Precision was **64.71%** in the June–October subset, where recall fell to **6.88%**.

The test split was excluded from feature selection, fitting and threshold selection. For date D, prediction selects a snapshot whose last eligible training date is strictly before D. Before the first snapshot, the volatility and earnings risk score drives the decision. If a practice row has masked implied volatility, prediction can use the most recent supplied earlier volatility. Other unavailable regression inputs map to the fitted training median. At runtime the program uses the supplied CSVs and the committed coefficients with no network requests.

## Reproduce

```sh
python -m pip install -r requirements.txt
python prepare_user_historical_info.py
python fit_model.py
python evaluator/run_evaluation.py .
```

Python 3.12 or later is required to rebuild the fitted model. The committed feature CSV and coefficients support offline evaluation, including after historical hourly bars age out of the source API.
