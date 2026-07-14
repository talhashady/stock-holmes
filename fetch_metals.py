import urllib.request
import json
import sys
import os

def load_env():
    # Helper to load .env manually if present
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

def fetch_via_alpha_vantage(api_key):
    """Attempts to fetch gold & silver spot prices via Alpha Vantage."""
    def query_symbol(symbol):
        url = f"https://www.alphavantage.co/query?function=GOLD_SILVER_SPOT&symbol={symbol}&apikey={api_key}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            if "Information" in data:
                raise ValueError(f"Rate limited: {data['Information']}")
            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            return data

    print("Attempting to fetch spot prices via Alpha Vantage...")
    # Fetch Gold
    gold = query_symbol("GOLD")
    
    # Simple delay for free key limits
    import time
    time.sleep(1.5)
    
    # Fetch Silver
    silver = query_symbol("SILVER")
    
    return {
        "gold": gold,
        "silver": silver
    }

def fetch_via_twelve_data(api_key):
    """Attempts to fetch gold spot price via Twelve Data."""
    print("Attempting to fetch spot prices via Twelve Data...")
    url = f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={api_key}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req, timeout=10) as response:
        data = json.loads(response.read().decode('utf-8'))
        if isinstance(data, dict) and data.get("status") == "error":
            raise ValueError(f"API Error: {data.get('message')}")
            
        price = data.get("price")
        if not price:
            raise ValueError("No price field returned in response.")
            
        return {
            "nominal": "XAUUSD",
            "price": price
        }

def main():
    load_env()
    
    # Parse inputs / fallbacks
    alpha_key = "demo"
    if len(sys.argv) > 1:
        # Check if the passed key is Twelve Data (32 chars) vs Alpha Vantage (16 chars)
        passed_key = sys.argv[1]
        if len(passed_key) == 16:
            alpha_key = passed_key
        elif len(passed_key) == 32:
            os.environ["TWELVE_DATA_API_KEY"] = passed_key
            
    # Try Alpha Vantage first (contains both Gold and Silver)
    av_success = False
    try:
        av_data = fetch_via_alpha_vantage(alpha_key)
        print("\n=== Success: Alpha Vantage Spot Price Data ===")
        print("Gold Spot Price:")
        print(json.dumps(av_data["gold"], indent=2))
        print("Silver Spot Price:")
        print(json.dumps(av_data["silver"], indent=2))
        av_success = True
    except Exception as e:
        print(f"Alpha Vantage request failed: {e}")
        
    # Failover to Twelve Data if Alpha Vantage failed
    if not av_success:
        print("\n[FAILOVER] Shifting to Twelve Data API for Gold...")
        td_key = os.getenv("TWELVE_DATA_API_KEY", "")
        if not td_key:
            print("Error: Failover failed because TWELVE_DATA_API_KEY is not configured in .env or system environment.")
            sys.exit(1)
            
        try:
            td_data = fetch_via_twelve_data(td_key)
            print("\n=== Success: Twelve Data Spot Price Data ===")
            print(json.dumps(td_data, indent=2))
            print("Note: Spot Silver (XAG/USD) is not available on the Twelve Data free tier plan.")
        except Exception as td_err:
            print(f"Twelve Data request also failed: {td_err}")
            sys.exit(1)

if __name__ == "__main__":
    main()
