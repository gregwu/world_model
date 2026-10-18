"""Export daily bars, daily indicators and regular-session 5m bars from Postgres to Parquet.

Reads DATABASE_URL from .env. Sources:
  ohlcv_daily       -> data/daily.parquet
  indicators_daily  -> data/indicators_daily.parquet
  yahoo5m (*.US)    -> data/intraday_5m.parquet  (09:30-16:00 ET only; extended-hours bars carry ~no volume)
"""
from __future__ import annotations
import argparse, io, subprocess
from pathlib import Path
import pandas as pd

DEFAULT_TICKERS = ("SPY,QQQ,IWM,AAPL,MSFT,NVDA,TSLA,AMZN,META,GOOGL,"
                   "AMD,AVGO,JPM,BAC,XOM,NFLX,COST,UNH,LLY,WMT")


def database_url(env_path: str = ".env") -> str:
    for line in open(env_path):
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError(f"DATABASE_URL not found in {env_path}")


def query(url: str, sql: str) -> pd.DataFrame:
    out = subprocess.run(["psql", url, "-c", f"\\copy ({sql}) to stdout with csv header"],
                         check=True, capture_output=True, text=True).stdout
    return pd.read_csv(io.StringIO(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=DEFAULT_TICKERS)
    ap.add_argument("--daily-start", default="2024-01-01")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()
    url = database_url(); out = Path(args.out_dir); out.mkdir(exist_ok=True)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    plain = ",".join(f"'{t}'" for t in tickers)
    dotted = ",".join(f"'{t}.US'" for t in tickers)

    d = query(url, f"select ticker as symbol, date as timestamp, open, high, low, close, volume "
                   f"from ohlcv_daily where ticker in ({plain}) and date >= '{args.daily_start}' order by 1,2")
    d["timestamp"] = pd.to_datetime(d["timestamp"]).dt.tz_localize("America/New_York")
    d.to_parquet(out / "daily.parquet", index=False)

    ind = query(url, f"select * from indicators_daily where ticker in ({plain}) "
                     f"and date >= '{args.daily_start}' order by ticker, date")
    ind = ind.rename(columns={"ticker": "symbol"})
    ind.to_parquet(out / "indicators_daily.parquet", index=False)

    i = query(url, f"select replace(ticker,'.US','') as symbol, ts as timestamp, open_price as open, "
                   f"high_price as high, low_price as low, close_price as close, volume "
                   f"from yahoo5m where ticker in ({dotted}) order by 1,2")
    et = pd.to_datetime(i["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.time
    i = i[(et >= pd.Timestamp("09:30").time()) & (et < pd.Timestamp("16:00").time())]
    i.to_parquet(out / "intraday_5m.parquet", index=False)

    print(f"daily {d.shape} {d.timestamp.min()} -> {d.timestamp.max()}")
    print(f"indicators {ind.shape} {ind.date.min()} -> {ind.date.max()}")
    print(f"5m {i.shape} {i.timestamp.min()} -> {i.timestamp.max()}")
    print(i.groupby("symbol").size().to_string())
    missing = sorted(set(tickers) - set(i.symbol) | set(tickers) - set(d.symbol))
    if missing: print("missing in 5m or daily:", missing)


if __name__ == "__main__":
    main()
