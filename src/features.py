import numpy as np
import pandas as pd

def _ensure_close(df: pd.DataFrame) -> pd.DataFrame:
    """Use adjusted close if available, else close price."""
    if "Adj Close" in df.columns:
        return pd.to_numeric(df["Adj Close"], errors="coerce")
    return pd.to_numeric(df["Close"], errors="coerce")

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)

def _macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return (macd_line - signal_line).fillna(0.0)

