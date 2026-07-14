import urllib.request
import json
import sys

def fetch_spot_price(symbol, api_key="demo"):
    url = f"https://www.alphavantage.co/query?function=GOLD_SILVER_SPOT&symbol={symbol}&apikey={api_key}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            
            # Check for API error or message
            if "Information" in data:
                print(f"Alpha Vantage Note: {data['Information']}")
                return None
            if "Error Message" in data:
                print(f"API Error for {symbol}: {data['Error Message']}")
                return None
                
            return data
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def main():
    api_key = "demo"
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
        
    print(f"Fetching gold & silver spot prices from Alpha Vantage (using key: {'***' if api_key != 'demo' else 'demo'})...\n")
    
    # Fetch Gold
    gold_data = fetch_spot_price("GOLD", api_key)
    if gold_data:
        print("Gold Spot Price Data:")
        print(json.dumps(gold_data, indent=2))
        print()
        
    # Rate limit buffer for free API keys
    import time
    time.sleep(1.5)
        
    # Fetch Silver
    silver_data = fetch_spot_price("SILVER", api_key)
    if silver_data:
        print("Silver Spot Price Data:")
        print(json.dumps(silver_data, indent=2))
        print()

if __name__ == "__main__":
    main()
