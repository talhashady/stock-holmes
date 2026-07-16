import os
import time
import logging
import pandas as pd
from typing import Optional
from twelvedata import TDClient

from src.serving.db_utils import (
    init_db, save_candles, get_latest_candle_time, init_cross_asset_tables
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ingestion.fetcher")

# Helper to load .env manually if present
def load_env():
    # .env path relative to fetcher.py: 3 levels up
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

load_env()
DEFAULT_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")

def get_client(api_key: str = DEFAULT_API_KEY) -> TDClient:
    """Returns an initialized TDClient."""
    if not api_key:
        raise ValueError("Twelve Data API Key not configured.")
    return TDClient(apikey=api_key)

def fetch_candles(
    symbol: str = "XAU/USD",
    interval: str = "1min",
    outputsize: int = 100,
    api_key: str = DEFAULT_API_KEY,
    max_retries: int = 3,
    backoff_factor: int = 2
) -> Optional[pd.DataFrame]:
    """
    Fetches raw OHLCV candles from Twelve Data with retry and backoff logic.
    Converts and cleans column types.
    """
    client = get_client(api_key)
    retries = 0
    
    while retries < max_retries:
        try:
            logger.info(f"Requesting {outputsize} candles for {symbol} ({interval})...")
            ts = client.time_series(symbol=symbol, interval=interval, outputsize=outputsize)
            
            # Retrieve as Pandas DataFrame
            df = ts.as_pandas()
            
            if df is None or df.empty:
                logger.warning("Empty response from Twelve Data API.")
                return None
                
            # Clean indices and structure
            df = df.reset_index()
            # Twelve Data client uses 'datetime' as the timestamp column
            if "datetime" in df.columns:
                df = df.rename(columns={"datetime": "timestamp"})
                
            # Convert columns to correct datatypes
            df["timestamp"] = df["timestamp"].astype(str)
            if "volume" not in df.columns:
                df["volume"] = 0.0
                
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
            # Drop any row with NaNs in primary fields
            df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
            
            # Sort chronologically (Twelve Data returns new-to-old by default)
            df = df.sort_values(by="timestamp", ascending=True)
            
            logger.info(f"Successfully fetched {len(df)} candles.")
            return df
            
        except Exception as e:
            retries += 1
            sleep_time = backoff_factor ** retries
            clean_error = redact_secrets(str(e), api_key)
            logger.error(f"API Error (Attempt {retries}/{max_retries}): {clean_error}. Retrying in {sleep_time}s...")
            time.sleep(sleep_time)
            
    logger.error("Failed to fetch data after max retries.")
    return None

def redact_secrets(msg: str, key: str) -> str:
    """Removes sensitive API keys from log messages."""
    if not key or len(key) < 5:
        return msg
    return msg.replace(key, "********")

def fetch_and_cache(
    symbol: str = "XAU/USD",
    interval: str = "1min",
    api_key: str = DEFAULT_API_KEY,
    force_backfill_size: int = 5000,
    table_name: str = "candles"
) -> int:
    """
    Fetches candles incrementally to conserve API limits.
    If database is empty, performs a larger historical fetch (force_backfill_size).
    Else, dynamically computes elapsed time since last cached candle and requests
    only enough candles to fill the gap.
    
    Args:
        symbol: Twelve Data symbol (e.g. 'XAU/USD', 'DXY', 'USD/JPY').
        interval: Candle interval.
        api_key: Twelve Data API key.
        force_backfill_size: Number of candles for initial backfill.
        table_name: SQLite table to store candles in ('candles', 'candles_dxy', 'candles_usdjpy').
    """
    init_db()
    latest_time = get_latest_candle_time(table_name=table_name)
    
    if latest_time is None:
        logger.info(f"[{table_name}] Local database is empty. Performing initial historical backfill for {symbol}...")
        df = fetch_candles(symbol=symbol, interval=interval, outputsize=force_backfill_size, api_key=api_key)
    else:
        logger.info(f"[{table_name}] Database contains candles up to {latest_time}. Running incremental fetch for {symbol}...")
        try:
            latest_dt = pd.to_datetime(latest_time)
            now_dt = pd.Timestamp.now(tz="UTC").tz_localize(None)
            elapsed_minutes = int((now_dt - latest_dt).total_seconds() / 60)
            # Request at least 100, up to 5000 candles to cover the gap with a 10 candle buffer
            outputsize = max(100, min(5000, elapsed_minutes + 10))
            logger.info(f"Time elapsed since last sync: {elapsed_minutes} minutes. Querying outputsize: {outputsize}")
        except Exception as ex:
            logger.warning(f"Error calculating dynamic gap: {ex}. Defaulting outputsize to 100")
            outputsize = 100
            
        df = fetch_candles(symbol=symbol, interval=interval, outputsize=outputsize, api_key=api_key)
        
    if df is not None and not df.empty:
        inserted = save_candles(df, table_name=table_name)
        logger.info(f"[{table_name}] Saved {inserted} new candles to SQLite database.")
        return inserted
        
    return 0


# Cross-asset symbol → SQLite table mapping
# Note: DXY (US Dollar Index) is not available on Twelve Data as a tradable symbol.
# EUR/USD is used as a USD-strength proxy instead — EUR constitutes ~57.6% of the
# DXY basket, making it the strongest single-pair proxy for dollar strength.
CROSS_ASSET_SYMBOLS = {
    "XAU/USD": "candles",
    "EUR/USD": "candles_eurusd",
    "USD/JPY": "candles_usdjpy",
}

def fetch_and_cache_multi(
    symbols: dict = None,
    interval: str = "1min",
    api_key: str = DEFAULT_API_KEY,
    force_backfill_size: int = 5000,
    inter_call_delay: float = 1.5
) -> dict:
    """
    Fetches candles for multiple symbols sequentially, respecting API rate limits.
    Each symbol is stored in its own SQLite table.
    
    Cross-asset symbols (DXY, USDJPY) are non-critical: if their fetch fails,
    a warning is logged and execution continues. Only the primary XAUUSD fetch
    failure is propagated.
    
    Args:
        symbols: Dict mapping symbol name → table name. Defaults to CROSS_ASSET_SYMBOLS.
        interval: Candle interval (e.g. '1min', '5min').
        api_key: Twelve Data API key.
        force_backfill_size: Number of candles for initial backfill per symbol.
        inter_call_delay: Seconds to wait between API calls for rate-limit compliance.
    
    Returns:
        Dict mapping symbol name → number of inserted candles (or -1 on failure).
    """
    if symbols is None:
        symbols = CROSS_ASSET_SYMBOLS
    
    results = {}
    
    for i, (symbol, table_name) in enumerate(symbols.items()):
        # Rate-limit delay between calls (skip delay before first call)
        if i > 0:
            logger.info(f"Rate-limit delay: waiting {inter_call_delay}s before fetching {symbol}...")
            time.sleep(inter_call_delay)
        
        try:
            inserted = fetch_and_cache(
                symbol=symbol,
                interval=interval,
                api_key=api_key,
                force_backfill_size=force_backfill_size,
                table_name=table_name
            )
            results[symbol] = inserted
            logger.info(f"✅ {symbol} → {table_name}: {inserted} new candles cached.")
        except Exception as e:
            results[symbol] = -1
            if table_name == "candles":
                # Primary symbol failure is critical
                logger.error(f"❌ CRITICAL: Failed to fetch primary symbol {symbol}: {e}")
                raise
            else:
                # Cross-asset failures are non-critical — model gracefully degrades
                logger.warning(f"⚠️ Non-critical: Failed to fetch {symbol}: {e}. "
                             f"Cross-asset correlation features will be NaN for this cycle.")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest XAU/USD + cross-asset market data.")
    parser.add_argument("--key", type=str, default=DEFAULT_API_KEY, help="Twelve Data API key.")
    parser.add_argument("--backfill", type=int, default=5000, help="Initial backfill candle count.")
    parser.add_argument("--multi", action="store_true", default=True,
                        help="Fetch all cross-asset symbols (DXY, USDJPY) alongside XAUUSD.")
    args = parser.parse_args()
    
    if args.multi:
        results = fetch_and_cache_multi(api_key=args.key, force_backfill_size=args.backfill)
        logger.info(f"Multi-symbol ingestion complete: {results}")
    else:
        fetch_and_cache(api_key=args.key, force_backfill_size=args.backfill)
