"""Pre-open signals from India equity funds and US depositary receipts."""
from bisect import bisect_left
from datetime import datetime, time, timedelta, timezone
from statistics import median
from math import log


GAP_MARKETS = {
    "tokyo_fund": "1678.T", "hk_fund": "2836.HK", "yen": "JPY=X",
    "hk_dollar": "HKD=X", "rupee": "INR=X", "nifty_hourly": "^NSEI",
}
ADR_SYMBOLS = {
    "DRREDDY": "RDY", "HDFCBANK": "HDB", "ICICIBANK": "IBN",
    "INFY": "INFY", "WIPRO": "WIT",
}


def adr_session_move(history, previous_india, day):
    """From the first US open after India's close to the last completed US close."""
    first = bisect_left(history.days, previous_india)
    last = bisect_left(history.days, day) - 1
    if first > last:
        return None
    open_ = history.opens[first]
    return 100 * log(history.closes[last] / open_) if open_ and open_ > 0 else None


def estimate_gap(day, nifty, hourly):
    """Compare morning ETF/FX quotes to parity observed during overlapping sessions."""
    cutoff = datetime.combine(day, time(3, 35), timezone.utc)
    last = nifty.before(day)

    def rupee_price(fund, fx, at, max_age):
        price = hourly[fund].close_at(at, max_age)
        local_per_usd = hourly[fx].close_at(at, timedelta(hours=2))
        inr_per_usd = hourly["rupee"].close_at(at, timedelta(hours=2))
        if price and local_per_usd and inr_per_usd:
            return price * inr_per_usd / local_per_usd
        return None

    estimates = []
    for fund, fx, overlap in (("tokyo_fund", "yen", time(6)), ("hk_fund", "hk_dollar", time(7, 45))):
        current = rupee_price(fund, fx, cutoff, timedelta(minutes=95))
        if current is None:
            continue
        ratios = []
        for session in reversed(nifty.days[max(0, last - 5):last + 1]):
            at = datetime.combine(session, overlap, timezone.utc)
            fund_price = rupee_price(fund, fx, at, timedelta(minutes=50))
            index = hourly["nifty_hourly"].close_at(at, timedelta(minutes=30))
            if fund_price and index:
                ratios.append(index / fund_price)
                if len(ratios) == 3:
                    break
        if ratios:
            estimates.append(100 * log(current * median(ratios) / nifty.closes[last]))
    return median(estimates) if estimates else None
