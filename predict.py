#!/usr/bin/env python3
"""Flag Nifty-50 stocks at risk of crossing their previous-close 2% band."""

import argparse
from bisect import bisect_left
import csv
import json
import math
from pathlib import Path
from statistics import median


EVENT_FEATURES = ("earnings_released", "earnings_scheduled")
MODEL_FEATURES = (
    "iv", "iv_relative", "range", "turnover_surprise", *EVENT_FEATURES,
    "india_vix", "us_vix", "es_abs_overnight", "oil_abs_overnight",
    "nikkei_abs_preopen", "hang_seng_abs_preopen", "india_etf_abs_session",
)
PRIOR_WEIGHT = 0.35
MAX_STANDARDIZED_FEATURE = 6.0


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def log_scale(value, anchor=1.0):
    return math.log(value / anchor) if value is not None and value > 0 else None


def positive(row, key):
    value = number(row.get(key))
    return value if value is not None and value > 0 else None


def absolute(row, key):
    value = number(row.get(key))
    return min(abs(value), 10.0) if value is not None else None


def daily_features(rows, market, prior_iv=None):
    """Build features from the supplied pre-open rows and a strict-prior fallback."""
    prior_iv = prior_iv or {}
    ivs = [positive(row, "vix") or prior_iv.get(row["name"]) for row in rows]
    available = [value for value in ivs if value is not None]
    market_iv = median(available) if available else None
    result = []
    for row, iv in zip(rows, ivs, strict=True):
        close, high, low = (positive(row, col) for col in ("pclose", "phi", "plo"))
        turnover, turnover_20 = (positive(row, col) for col in ("pvolume", "mddv_20"))
        good_range = close and high and low and low <= close <= high
        previous_range = min(100 * (high - low) / close, 20) if good_range else None
        surprise = math.log(turnover / turnover_20) if turnover and turnover_20 else None
        stock_market = market.get(row["name"], {})
        result.append({
            "iv": log_scale(iv, 28),
            "iv_relative": log_scale(iv, market_iv) if market_iv else None,
            "range": previous_range,
            "turnover_surprise": surprise,
            "india_vix": log_scale(positive(stock_market, "india_vix"), 16),
            "us_vix": log_scale(positive(stock_market, "us_vix"), 20),
            "sp500_abs_return": absolute(stock_market, "sp500_return"),
            "india_etf_abs_session": absolute(stock_market, "india_etf_session"),
            "es_abs_overnight": absolute(stock_market, "es_overnight"),
            "oil_abs_overnight": absolute(stock_market, "oil_overnight"),
            "nikkei_abs_preopen": absolute(stock_market, "nikkei_preopen"),
            "hang_seng_abs_preopen": absolute(stock_market, "hang_seng_preopen"),
            "earnings_released": number(stock_market.get("earnings_released")),
            "earnings_scheduled": number(stock_market.get("earnings_scheduled")),
        })
    return result


def previous_implied_volatility(directory, day, missing):
    suffix = "_n50.info.csv"
    ivs = {}
    for path in sorted(directory.glob(f"*{suffix}"), reverse=True):
        if path.name.removesuffix(suffix) >= day:
            continue
        for row in read_csv(path):
            name = row["name"]
            if name in missing and (value := positive(row, "vix")) is not None:
                ivs[name] = value
                missing.remove(name)
        if not missing:
            break
    return ivs


def prior_risk(features):
    """Conservative cold-start risk from implied volatility and a market proxy."""
    overnight = (
        0.50 * (features["es_abs_overnight"] or 0.0)
        + 0.18 * (features["oil_abs_overnight"] or 0.0)
        + 0.27 * (features["india_etf_abs_session"] or 0.0)
        + 0.18 * (features["nikkei_abs_preopen"] or 0.0)
        + 0.12 * (features["hang_seng_abs_preopen"] or 0.0)
    )
    return (
        -0.85 + 3.7 * (features["iv"] or 0.0)
        + 0.65 * (features["india_vix"] or 0.0)
        + 0.12 * (features["sp500_abs_return"] or 0.0)
        + 0.65 * overnight
        + 1.45 * (features["earnings_released"] or 0.0)
        + 0.35 * (features["earnings_scheduled"] or 0.0)
    )


def fitted_risk(features, name, snapshot):
    score = snapshot["intercept"] + snapshot["stock_effects"].get(name, 0.0)
    for key, center, scale, weight in zip(
        MODEL_FEATURES, snapshot["centers"], snapshot["scales"], snapshot["weights"], strict=True
    ):
        if features[key] is not None:
            z = (features[key] - center) / scale
            score += weight * max(-MAX_STANDARDIZED_FEATURE, min(z, MAX_STANDARDIZED_FEATURE))
    return (1 - PRIOR_WEIGHT) * score + PRIOR_WEIGHT * prior_risk(features)


def predict_day(rows, market, prior_iv, day, model):
    snapshots = model["snapshots"]
    index = bisect_left([snapshot["as_of"] for snapshot in snapshots], day) - 1
    snapshot = snapshots[index] if index >= 0 else None
    features = daily_features(rows, market, prior_iv)
    scores = (
        [prior_risk(values) for values in features] if snapshot is None else
        [fitted_risk(values, row["name"], snapshot) for row, values in zip(rows, features, strict=True)]
    )
    return [score >= model["threshold"] for score in scores]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--historical-info", type=Path, required=True)
    parser.add_argument("--user-historical-info", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_csv(args.historical_info / f"{args.date}_n50.info.csv")
    market = {}
    if args.user_historical_info:
        with args.user_historical_info.open(newline="", encoding="utf-8") as stream:
            market = {row["name"]: row for row in csv.DictReader(stream) if row["date"] == args.date}
    missing_iv = {row["name"] for row in rows if positive(row, "vix") is None}
    prior_iv = previous_implied_volatility(args.historical_info, args.date, missing_iv) if missing_iv else {}

    model = json.loads(Path(__file__).with_name("model.json").read_text(encoding="utf-8"))
    predictions = predict_day(rows, market, prior_iv, args.date, model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / f"{args.date}.predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("name", "prediction"))
        writer.writerows((row["name"], flag) for row, flag in zip(rows, predictions, strict=True))


if __name__ == "__main__":
    main()
