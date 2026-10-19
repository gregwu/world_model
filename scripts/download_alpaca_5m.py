"""Download historical 5-minute bars from Alpaca (SIP feed, split/dividend adjusted), regular session only.

Reads ALPACA_API_KEY / ALPACA_SECRET_KEY from .env. Writes one Parquet file per symbol
(symbol,timestamp,open,high,low,close,volume) to --out-dir; existing files are skipped, so reruns resume.
pandas reads the whole directory as one table, so the directory can be used directly as data.intraday_path.

Usage:
  python scripts/download_alpaca_5m.py --top-liquid 100 --out-dir data/alpaca5m/bars
  python scripts/download_alpaca_5m.py --tickers SPY,QQQ --start 2016-01-01
  python scripts/download_alpaca_5m.py --tickers-file data/alpaca5m/sp500_tickers.txt --top-liquid 100 --workers 8

Requests are mostly network-bound (~2.5 s for a 10k-bar page), so --workers runs symbols in parallel while the
shared limiter keeps the total under --per-minute. The free plan rejects SIP queries that reach the last 15 minutes,
so --end defaults to two days ago.
"""
from __future__ import annotations
import argparse, json, threading, time, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

URL = "https://data.alpaca.markets/v2/stocks/bars"


def read_env(path: str = ".env") -> dict:
    env = {}
    for line in open(path):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"')
    return env


class RateLimiter:
    """Thread-safe: spaces request starts at least 60/per_minute seconds apart across all workers."""

    def __init__(self, per_minute: int):
        self.interval = 60.0 / per_minute; self.next_slot = 0.0; self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic(); slot = max(now, self.next_slot); self.next_slot = slot + self.interval
        if slot > now: time.sleep(slot - now)


def fetch_symbol(symbol, start, end, headers, limiter, feed="sip"):
    rows, token = [], None
    while True:
        params = {"symbols": symbol, "timeframe": "5Min", "start": start, "end": end, "limit": 10000,
                  "adjustment": "all", "feed": feed, "sort": "asc"}
        if token: params["page_token"] = token
        req = urllib.request.Request(URL + "?" + urllib.parse.urlencode(params), headers=headers)
        for attempt in range(6):
            limiter.wait()
            try:
                data = json.load(urllib.request.urlopen(req, timeout=60)); break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(2 ** attempt * 5); continue
                raise
            except (urllib.error.URLError, TimeoutError):
                if attempt < 5: time.sleep(2 ** attempt * 5); continue
                raise
        rows.extend((data.get("bars") or {}).get(symbol, []))
        token = data.get("next_page_token")
        if not token: return rows


def to_frame(symbol, rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty: return df
    df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    et = df["timestamp"].dt.tz_convert("America/New_York").dt.time
    df = df[(et >= pd.Timestamp("09:30").time()) & (et < pd.Timestamp("16:00").time())]
    df.insert(0, "symbol", symbol)
    return df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="")
    ap.add_argument("--top-liquid", type=int, default=0, help="top N of ohlcv_daily by median 2014 dollar volume")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=(pd.Timestamp.now("UTC") - pd.Timedelta(days=2)).strftime("%Y-%m-%d"))
    ap.add_argument("--tickers-file", default="", help="one symbol per line; combined with --tickers/--top-liquid")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--keys", default="live", help="comma list of key pairs from .env: live (ALPACA_API_KEY) and/or "
                    "paper (ALPACA_API_KEY_PAPER); symbols alternate between them. Measured 2026-10-19: the live and paper keys "
                    "share ONE rate limit (a second key at 180/min each gave 429s), so this does not add throughput.")
    ap.add_argument("--out-dir", default="data/alpaca5m/bars")
    ap.add_argument("--per-minute", type=int, default=180)
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.tickers_file:
        tickers += [t.strip().upper() for t in open(args.tickers_file) if t.strip()]
    if args.top_liquid:
        from export_from_db import database_url, query
        top = query(database_url(), "select ticker from ohlcv_daily where date between '2014-01-01' and '2014-12-31' "
                                    "and close >= 5 group by ticker having count(*) >= 200 order by "
                                    f"percentile_cont(0.5) within group (order by close*volume) desc limit {args.top_liquid}")
        tickers += top["ticker"].str.upper().tolist()
    tickers = list(dict.fromkeys(tickers))  # de-duplicate, keep order
    env = read_env(); suffix = {"live": "", "paper": "_PAPER"}
    # Pages hold ~2.6k 5-minute bars (the limit appears to count 1-minute source bars), so a symbol takes ~130
    # requests and the run is bound by the 200/min rate limit (~0.6 min per symbol at 180/min).
    clients = [({"APCA-API-KEY-ID": env[f"ALPACA_API_KEY{suffix[k]}"], "APCA-API-SECRET-KEY": env[f"ALPACA_SECRET_KEY{suffix[k]}"]},
                RateLimiter(args.per_minute)) for k in args.keys.split(",")]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tickers if not (out / f"{t.replace('/', '_')}.parquet").exists()]
    print(f"{len(tickers)} symbols, {len(tickers) - len(todo)} already downloaded, {len(todo)} to fetch "
          f"with {args.workers} workers and {len(clients)} key(s)", flush=True)
    t0 = time.time(); empty, failed = [], []

    def one(sym):
        headers, limiter = clients[todo.index(sym) % len(clients)]
        df = to_frame(sym, fetch_symbol(sym, args.start, args.end, headers, limiter))
        if not df.empty:
            tmp = out / f".{sym.replace('/', '_')}.parquet.tmp"
            df.to_parquet(tmp, index=False); tmp.rename(out / f"{sym.replace('/', '_')}.parquet")  # atomic: no partial files
        return df

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, sym): sym for sym in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            sym = futures[fut]
            try:
                df = fut.result()
            except Exception as e:  # keep going; rerun to retry failures
                failed.append(sym); print(f"[{k}/{len(todo)}] {sym}: FAILED {e}", flush=True); continue
            if df.empty:
                empty.append(sym); print(f"[{k}/{len(todo)}] {sym}: no bars", flush=True); continue
            print(f"[{k}/{len(todo)}] {sym}: {len(df):,} bars {df.timestamp.min():%Y-%m-%d} -> {df.timestamp.max():%Y-%m-%d} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min; no data: {empty}; failed (rerun to retry): {failed}", flush=True)


if __name__ == "__main__":
    main()
