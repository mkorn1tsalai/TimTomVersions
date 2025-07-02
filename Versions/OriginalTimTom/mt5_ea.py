import MetaTrader5 as mt5
import time
import datetime as dt
import pytz
import os

# MT5 initialization
def initialize_mt5(path):
    if not mt5.initialize(path=path):
        print("initialize() failed. Retrying in 5 seconds...",mt5.last_error())
        time.sleep(5)
        initialize_mt5(path)

# Path to the MetaTrader 5 terminal
fpath = "C:\\Program Files\\MetaTrader5_dummy\\terminal64.exe"


initialize_mt5(path = fpath )

symbol = 'EURUSD'
timeframe = mt5.TIMEFRAME_M15

# Define your timezone (e.g., "America/New_York" for Eastern Time)
local_timezone = pytz.timezone("America/New_York")

run_bot = False
while True:
    while True:
        try:
            current_bar_time = mt5.copy_rates_from_pos(symbol, timeframe, 1, 1)[0][0]
            if current_bar_time is not None:
                break
        except TypeError:
            pass
        time.sleep(1)
  
    next_bar_time = dt.datetime.fromtimestamp(current_bar_time, pytz.utc).astimezone(local_timezone)
    #next_bar_time = dt.datetime.fromtimestamp(current_bar_time, pytz.timezone("Etc/GMT"))
    next_bar_time = next_bar_time.replace(second=0, microsecond=0)
 
    while True:
        now = dt.datetime.now(pytz.utc).astimezone(local_timezone)
        #now = dt.datetime.now(pytz.timezone("Etc/GMT+5"))
        #utc_minus_5_time = now.astimezone(pytz.timezone("Etc/GMT+5"))
        time_until_next_bar = (next_bar_time - now).total_seconds()
        print(now)
 

        if now.minute in [0, 15, 30, 45]:
            run_bot = True
            break

        print("Waiting for next bar...")
        time.sleep(60)

    if run_bot:
        # Check if it's Friday after 5 PM
        if now.weekday() == 4 and now.hour >= 17:
            weekend_switch = True
        elif now.weekday() == 5:
            weekend_switch = True
        elif now.weekday() == 6 and now.hour < 17:
            weekend_switch = True
        else:
            weekend_switch = False
        while weekend_switch:
            print("markets closed let TIM-TOM sleep...")
            break
        else:
            print("markets open let TIM-TOM loose...")
            os.chdir('C:/Users/Administrator/Desktop/fred-manager')
            try:
                with open('fred-deploy.py') as f:
                    code = compile(f.read(), 'fred-deploy.py', 'exec')
                    exec(code)
            except Exception as e:
                print("Error occurred during bot execution:", e)
            finally:
                f.close()

        # Reset run_bot flag
        run_bot = False

    time.sleep(60)

mt5.shutdown()
