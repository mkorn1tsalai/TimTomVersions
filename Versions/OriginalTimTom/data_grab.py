import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz
import os

# --- Setup
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M1  # 1-minute data
timezone = pytz.timezone("Etc/UTC")
from_date = datetime(2000, 1, 1, tzinfo=timezone)
to_date = datetime(2024, 12, 31, tzinfo=timezone)

# --- Initialize MT5 connection
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()

# --- Download 1-minute data
rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)
mt5.shutdown()

# --- Check data
if rates is None or len(rates) == 0:
    print("No data retrieved.")
    quit()

# --- Create DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# --- Split 'time' into 'Date' and 'Time' columns
df['Date'] = df['time'].dt.strftime('%Y.%m.%d')
df['Time'] = df['time'].dt.strftime('%H:%M:%S')

# --- Reorder and rename columns to match target format
df = df[['Date', 'Time', 'open', 'high', 'low', 'close', 'tick_volume', 'real_volume', 'spread']]
df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread']

# --- Save to CSV
os.makedirs("data", exist_ok=True)
df.to_csv("data/itd_2000_2024.csv", index=False)

print("1-minute data export completed to: data/itd_2000_2024.csv")

