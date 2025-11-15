import os
import pandas as pd

def get_prices(
    symbol: str,
    start: str,
    end: str | None = None,
    cache_dir: str = "data",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download daily prices using yfinance and cache to CSV.
    Returns a DataFrame indexed by Date with standard OHLCV (+ Adj Close).
    """
    os.makedirs(cache_dir, exist_ok=True)
    csv_path = os.path.join(cache_dir, f"{symbol}_{start}_{end or 'latest'}.csv")

    if use_cache and os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")

    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    df.index.name = "Date"
    df.to_csv(csv_path)   # saves the index directly (no reset_index needed)
    return df