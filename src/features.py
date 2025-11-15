import numpy as np
import pandas as pd

# ----------------- helpers ----------------- #
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

# ----------------- main ----------------- #
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Features: 
    - Return_1d, Return_1d_lag1 , Return_5d, Return_21d
    - MA_20, Price_vs_MA_20
    - Vol_20, Vol_60, Vol_Ratio_20_60
    - RSI_14, MACD_Hist
    - BB_PctB_20_2, BB_Width_20_2
    - Target_Return_1d (shifted -1)
    """

    # robust index/column 
    df = df.copy().sort_index().reset_index 
    df = df[~df.index.duplicated(keep="first")]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # base price series
    close = _ensure_close(df)

    # returns short + multi-horizon 
    ret1 = close.pct_change(1)
    ret5 = close.pct_change(5)
    ret21 = close.pct_change(21)

    feat = pd.DataFrame(index=df.index)
    feat["Close"] = close
    feat["Return_1d"] = ret1
    feat["Return_1d_lag1"] = ret1.shift(1)
    feat["Return_5d"] = ret5
    feat["Return_21d"] = ret21

    # moving average

    