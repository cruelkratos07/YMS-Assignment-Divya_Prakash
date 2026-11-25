#!/usr/bin/env python3
"""Rebuild strictly dated model snapshots using the supplied training labels."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from predict import (
    EVENT_FEATURES, MAX_STANDARDIZED_FEATURE, MODEL_FEATURES,
    daily_features, fitted_risk, monthly_volatility, positive, read_csv,
)
from test_predictions import parse_bool, score


WARMUP_DATES = 8
REFIT_EVERY = 2
RIDGE_PENALTY = 100 / 3
STOCK_SCALE = 0.65
THRESHOLD = 0.40


def load_training(data, feature_csv):
    labels = {(r["date"], r["name"]): (float(r["max_move_pct"]), parse_bool(r["moved"]))
              for r in read_csv(data / "train_labels.csv")}
    date_blocks = sorted({date for date, _ in labels})
    markets = {day: {} for day in date_blocks}
    with feature_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["date"] in markets:
                markets[row["date"]][row["name"]] = row
    dates, names, features, complete = [], [], [], []
    prior_iv = {}
    for index, day in enumerate(date_blocks):
        rows = read_csv(data / "historical_info" / f"{day}_n50.info.csv")
        fresh_iv = {row["name"]: iv for row in rows
                    if (iv := positive(row, "vix")) is not None}
        complete_day = len(fresh_iv) >= 0.9 * len(rows)
        range_vol = monthly_volatility(data / "historical_info", day, rows)
        day_features = daily_features(rows, markets[day], prior_iv, range_vol)
        for row, values in zip(rows, day_features, strict=True):
            dates.append(index)
            names.append(row["name"])
            features.append(values)
            complete.append(complete_day)
        prior_iv.update(fresh_iv)

    target = [labels[date_blocks[index], name] for index, name in zip(dates, names, strict=True)]
    moves, binary = zip(*target)
    return (date_blocks, np.array(dates), np.array(names), features, np.array(complete),
            np.log(np.maximum(moves, 0.05)), np.array(binary))


def fit_snapshots(date_blocks, dates, names, features, complete, log_moves, target):
    x = np.array([[row[key] for key in MODEL_FEATURES] for row in features], dtype=float)
    symbols = np.unique(names)
    stock_vectors = STOCK_SCALE * (names[:, None] == symbols)
    snapshots = []
    validation = {}
    for first in range(WARMUP_DATES, len(date_blocks) + 1, REFIT_EVERY):
        fit_rows = (dates < first) & complete
        train_x = x[fit_rows]
        available = np.isfinite(train_x).any(axis=0)
        centers, scales = np.zeros(x.shape[1]), np.ones(x.shape[1])
        centers[available] = np.nanmedian(train_x[:, available], axis=0)
        scales[available] = np.maximum(np.nanstd(train_x[:, available], axis=0), 0.05)
        for j, key in enumerate(MODEL_FEATURES):
            if key in EVENT_FEATURES:
                scales[j] = 0.4
        standardized = np.clip(
            np.nan_to_num((x[fit_rows] - centers) / scales),
            -MAX_STANDARDIZED_FEATURE, MAX_STANDARDIZED_FEATURE,
        )
        model_input = np.column_stack((standardized, stock_vectors[fit_rows]))
        with threadpool_limits(limits=1):
            fit = Ridge(alpha=RIDGE_PENALTY).fit(model_input, log_moves[fit_rows])
        # Express distance from the 2% boundary in residual standard deviations.
        residuals = log_moves[fit_rows] - fit.predict(model_input)
        score_scale = 1.65 / max(float(np.std(residuals)), 1e-6)
        weights = fit.coef_ * score_scale
        snapshot = {
            "as_of": date_blocks[first - 1],
            "intercept": float((fit.intercept_ - np.log(2.0)) * score_scale),
            "centers": centers.tolist(),
            "scales": scales.tolist(),
            "weights": weights[:len(MODEL_FEATURES)].tolist(),
            "stock_effects": dict(zip(symbols.tolist(), (STOCK_SCALE * weights[len(MODEL_FEATURES):]).tolist())),
        }
        snapshots.append(snapshot)
        valid_rows = (dates >= first) & (dates < first + REFIT_EVERY) & complete
        for i in np.flatnonzero(valid_rows):
            key = (date_blocks[dates[i]], names[i])
            validation[key] = target[i], fitted_risk(features[i], names[i], snapshot) >= THRESHOLD
    return snapshots, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--market-info", type=Path, default=Path("user_historical_info.csv"))
    parser.add_argument("--out", type=Path, default=Path("model.json"))
    args = parser.parse_args()

    snapshots, validation = fit_snapshots(*load_training(args.data, args.market_info))
    artifact = {"threshold": THRESHOLD, "snapshots": snapshots}
    args.out.write_text(json.dumps(artifact, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")

    for label, rows in (
        ("Expanding-window validation", validation),
        ("Lower-prevalence period (June onward)", {k: v for k, v in validation.items() if k[0][4:6] >= "06"}),
    ):
        metrics = score({key: pair[0] for key, pair in rows.items()},
                        {key: pair[1] for key, pair in rows.items()})
        print(f"{label} ({metrics['rows']} rows): "
              f"precision {metrics['precision']:.4f}, recall {metrics['recall']:.4f}, "
              f"score {metrics['score']:.4f}")


if __name__ == "__main__":
    main()
