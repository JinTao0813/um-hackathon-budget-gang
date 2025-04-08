import requests
import pandas as pd
import time
from datetime import datetime, timedelta

API_KEY = "GlYxSZP9hnooNl6gGAjtkptkeqehSnk5C60Akhpw5zupBK6O"
BASE_URL = "https://api.datasource.cybotrade.rs/bybit-linear/candle"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
DAYS = 7
LIMIT = 1000  # Max per API call

end_time = int(time.time() * 1000)
start_time = end_time - DAYS * 24 * 60 * 60 * 1000

headers = {
    "X-API-KEY": API_KEY
}

all_candles = []

print(f"📡 Fetching {DAYS} days of {INTERVAL} data for {SYMBOL}")
while start_time < end_time:
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "start_time": start_time,
        "limit": LIMIT
    }

    response = requests.get(BASE_URL, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json().get("data", [])
        if not data:
            print("⚠️ No more data returned.")
            break
        all_candles.extend(data)
        print(f"✅ Retrieved {len(data)} candles. Total: {len(all_candles)}")
        last_time = data[-1]["start_time"]
        start_time = last_time + 60 * 1000  # next 1m
        time.sleep(0.5)  # Respect rate limits
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        break

df = pd.DataFrame(all_candles)
df["timestamp"] = pd.to_datetime(df["start_time"], unit="ms")
df.set_index("timestamp", inplace=True)
df = df[["open", "high", "low", "close", "volume"]]

filename = f"{SYMBOL}_{INTERVAL}_{DAYS}d.csv"
df.to_csv(filename)
print(f"\n✅ Saved {len(df)} rows to {filename}")
