#!/usr/bin/env python3
"""Build date/name rows with market features from completed daily closes."""

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, time as clock_time, timedelta, timezone
import json
import math
from pathlib import Path
import re
import time
from urllib.error import URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


SYMBOLS = {
    "nifty": "^NSEI",
    "india_vix": "^INDIAVIX",
    "sp500": "^GSPC",
    "us_vix": "^VIX",
    "india_etf": "INDY",
}
HOURLY_SYMBOLS = {"es": "ES=F", "oil": "BZ=F", "nikkei": "^N225", "hang_seng": "^HSI"}
RESULTS = re.compile(r"\b(?:financial|quarterly|audited|unaudited)\s+results\b", re.I)
EVENT_FEATURES = ("earnings_released", "earnings_scheduled")
FEATURES = (
    "india_vix", "sp500_return", "us_vix", "india_etf_session",
    "es_overnight", "oil_overnight", "nikkei_preopen", "hang_seng_preopen",
)


def nse_filings(endpoint, start, end):
    params = urlencode({
        "index": "equities",
        "from_date": start.strftime("%d-%m-%Y"),
        "to_date": end.strftime("%d-%m-%Y"),
    })
    request = Request(f"https://www.nseindia.com/api/{endpoint}?{params}", headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    })
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except (URLError, TimeoutError, ValueError) as exc:
            if attempt == 2:
                raise RuntimeError(f"Could not retrieve NSE {endpoint} for {start} to {end}") from exc
            time.sleep(attempt + 1)


class EarningsHistory:
    def __init__(self, names, start, end):
        self.releases = defaultdict(set)
        self.meetings = defaultdict(set)
        interval = timedelta(days=27)
        windows = []
        while start <= end:
            windows.append((start, min(start + interval, end)))
            start += interval + timedelta(days=1)

        def read_window(window):
            for row in nse_filings("corporate-announcements", *window):
                name = row["symbol"]
                subject = str(row.get("desc", "")).casefold()
                if name not in names or not row.get("exchdisstime"):
                    continue
                if subject in {"financial results", "integrated filing - financial"} or (
                    subject in {"outcome of board meeting", "press release"}
                    and RESULTS.search(str(row.get("attchmntText", "")))
                ):
                    self.releases[name].add(datetime.strptime(row["exchdisstime"], "%d-%b-%Y %H:%M:%S"))
            for row in nse_filings("corporate-board-meetings", *window):
                name = row["bm_symbol"]
                description = f'{row.get("bm_purpose", "")} {row.get("bm_desc", "")}'
                if name not in names or not row.get("sysTime") or not RESULTS.search(description):
                    continue
                meeting = datetime.strptime(row["bm_date"], "%d-%b-%Y").date()
                notice = datetime.strptime(row["sysTime"], "%d-%b-%Y %H:%M:%S")
                self.meetings[meeting, name].add(notice)

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(read_window, windows))

    def features(self, day, previous_india, name):
        # Exchange dissemination times are in India local time.
        cutoff = datetime.combine(day, clock_time(9, 15))
        previous_close = datetime.combine(previous_india, clock_time(15, 30))
        released = any(previous_close <= event < cutoff for event in self.releases[name])
        scheduled = any(notice < cutoff for notice in self.meetings[day, name])
        return int(released), int(scheduled)


def load_bars(symbol, start, end, interval):
    start_ts = datetime.combine(start, datetime.min.time(), timezone.utc)
    end_ts = datetime.combine(end, datetime.min.time(), timezone.utc)
    params = urlencode({
        "interval": interval,
        "period1": int(start_ts.timestamp()),
        "period2": int(end_ts.timestamp()),
    })
    ticker = quote(symbol, safe="")
    for attempt in range(3):
        host = "query1.finance.yahoo.com" if attempt < 2 else "query2.finance.yahoo.com"
        url = f"https://{host}/v8/finance/chart/{ticker}?{params}"
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=20) as response:
                result = json.load(response)["chart"]["result"][0]
            break
        except (URLError, TimeoutError, ValueError, TypeError) as exc:
            if attempt == 2:
                raise RuntimeError(f"Could not retrieve daily closes for {symbol}") from exc
            time.sleep(attempt + 1)

    return result


def daily_bars(symbol, start, end):
    result = load_bars(symbol, start, end, "1d")
    exchange_tz = ZoneInfo(result["meta"]["exchangeTimezoneName"])
    bars = {}
    quote_data = result["indicators"]["quote"][0]
    for ts, open_, close in zip(
        result["timestamp"], quote_data["open"], quote_data["close"], strict=True
    ):
        if close is not None and math.isfinite(close) and close > 0:
            bars[datetime.fromtimestamp(ts, exchange_tz).date()] = close, open_
    if len(bars) < 25:
        raise ValueError(f"Insufficient daily closes for {symbol}")
    return sorted(bars.items())


def log_return(after, before):
    return 100 * math.log(after / before)


class CloseHistory:
    def __init__(self, bars):
        self.days, prices = zip(*bars)
        self.closes, self.opens = zip(*prices)
        self.returns = (None,) + tuple(
            log_return(after, before)
            for before, after in zip(self.closes, self.closes[1:])
        )

    def before(self, day):
        # US and Indian closes bearing the prior exchange-local date were
        # complete before 09:15 IST. Skip stale values after extended closures.
        index = bisect_left(self.days, day) - 1
        if index < 0 or (day - self.days[index]).days > 7:
            raise ValueError(f"No recent completed market close before {day}")
        return index


class HourlyHistory:
    def __init__(self, symbol, start, end):
        result = load_bars(symbol, start, end, "60m")
        values = {}
        for ts, close in zip(
            result["timestamp"], result["indicators"]["quote"][0]["close"], strict=True
        ):
            if close is not None and math.isfinite(close) and close > 0:
                # Yahoo timestamps the start of the bar. Count the full hour
                # even when the exchange ends the session earlier.
                known_at = datetime.fromtimestamp(ts, timezone.utc) + timedelta(hours=1)
                values[known_at] = close
        self.times, self.closes = zip(*sorted(values.items()))

    def close_at(self, cutoff, max_age):
        i = bisect_right(self.times, cutoff) - 1
        if i < 0 or cutoff - self.times[i] > max_age:
            return None
        return self.closes[i]


def intraday_return(history, cutoff, anchor, session_only):
    max_current_age = timedelta(hours=3)
    max_anchor_age = timedelta(days=4) if session_only else timedelta(hours=3)
    close = history.close_at(cutoff, max_current_age)
    base = history.close_at(anchor, max_anchor_age)
    return log_return(close, base) if close is not None and base is not None else None


def market_features(day, histories, hourly):
    indices = {key: values.before(day) for key, values in histories.items()}
    us_etf = histories["india_etf"]
    etf_idx = indices["india_etf"]
    etf_open = us_etf.opens[etf_idx]
    # 03:35 UTC = 09:05 IST. Leave ten minutes for the bar to be published.
    cutoff = datetime.combine(day, clock_time(3, 35), timezone.utc)
    previous_india = histories["nifty"].days[indices["nifty"]]
    india_close = datetime.combine(previous_india, clock_time(10), timezone.utc)
    asia_anchor = datetime.combine(day, clock_time.min, timezone.utc) - timedelta(hours=15)
    return (
        histories["india_vix"].closes[indices["india_vix"]],
        histories["sp500"].returns[indices["sp500"]],
        histories["us_vix"].closes[indices["us_vix"]],
        log_return(us_etf.closes[etf_idx], etf_open) if etf_open and etf_open > 0 else None,
        intraday_return(hourly["es"], cutoff, india_close, False),
        intraday_return(hourly["oil"], cutoff, india_close, False),
        intraday_return(hourly["nikkei"], cutoff, asia_anchor, True),
        intraday_return(hourly["hang_seng"], cutoff, asia_anchor, True),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", type=Path, default=Path("data/all_dates.txt"))
    parser.add_argument("--names", type=Path, default=Path("data/n50_names.txt"))
    parser.add_argument("--out", type=Path, default=Path("user_historical_info.csv"))
    args = parser.parse_args()

    dates = sorted({datetime.strptime(day, "%Y%m%d").date()
                    for day in args.dates.read_text().split()})
    names = sorted(set(args.names.read_text().split()))
    if not dates or not names:
        parser.error("--dates and --names must be nonempty")

    start = dates[0] - timedelta(days=65)
    end = dates[-1] + timedelta(days=1)
    histories = {
        key: CloseHistory(daily_bars(symbol, start, end))
        for key, symbol in SYMBOLS.items()
    }
    hourly = {
        key: HourlyHistory(symbol, dates[0] - timedelta(days=8), end)
        for key, symbol in HOURLY_SYMBOLS.items()
    }
    earnings = EarningsHistory(set(names), dates[0] - timedelta(days=8), dates[-1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("date", "name", *FEATURES, *EVENT_FEATURES))
        for day in dates:
            values = tuple(f"{value:.6f}" if value is not None else ""
                           for value in market_features(day, histories, hourly))
            previous_india = histories["nifty"].days[histories["nifty"].before(day)]
            writer.writerows(
                (day.strftime("%Y%m%d"), name, *values, *earnings.features(day, previous_india, name))
                for name in names
            )
    print(f"Wrote {len(dates)} dates × {len(names)} stocks to {args.out}")


if __name__ == "__main__":
    main()
