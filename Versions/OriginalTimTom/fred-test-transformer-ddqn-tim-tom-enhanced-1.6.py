import os
import pandas as pd
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import backend as Add
from tensorflow.keras import backend as concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from keras.layers import LayerNormalization
import matplotlib.pyplot as plt
from datetime import datetime
import time
max_mem = 96*30
def test_model(epi, epi2, data, train, reward, reward2, agent_switch):
    # Use an absolute file path
    data = pd.read_csv(r'data\data_2023.csv')

    # Remove unnecessary columns (open, high, low, vol, spread)
    data = data[['Date', 'Time', 'TickVol', 'High', 'Low', 'Close']]

    # Parse Date column and set it as the index
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data.set_index('Datetime', inplace=True)
    data.drop(columns=['Date'], inplace=True)  # Drop Date column after creating Datetime index

    moving_averages = [96, 96*2, 96*3, 96*4, 96*5, 96*6]
    for index, window in enumerate(moving_averages):
        ma_column_name = f'MA_{index + 1}'
        data[ma_column_name] = data['Close'].rolling(window=window).mean()

    # Calculate RSI
    def calculate_rsi(data, window=96):
        delta = data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    data['RSI'] = calculate_rsi(data)

    # Calculate MACD
    short_window = 96
    long_window = 96*5
    signal_window = 96
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    data['MACD'] = data['Short_MA'] - data['Long_MA']
    data['Signal_Line'] = data['MACD'].rolling(window=signal_window).mean()

    # Calculate Bollinger Bands
    window = 96
    data['Rolling_Mean'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band'] = data['Rolling_Mean'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band'] = data['Rolling_Mean'] - 2 * data['Close'].rolling(window=window).std()

    # Calculate Stochastic Oscillator
    k_window = 96
    d_window = 24
    data['Lowest_Low'] = data['Low'].rolling(window=k_window).min()
    data['Highest_High'] = data['High'].rolling(window=k_window).max()
    data['%K'] = (data['Close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low']) * 100
    data['%D'] = data['%K'].rolling(window=d_window).mean()

    # Calculate Price Rate of Change (ROC)
    roc_window = 96
    data['ROC'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100

    # Calculate Average True Range (ATR)
    atr_window = 96
    data['TR'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR'] = data['TR'].rolling(window=atr_window).mean()

    # Calculate On-Balance Volume (OBV)
    data['Volume_Direction'] = data['TickVol'].apply(lambda x: 1 if x >= 0 else -1)
    data['OBV'] = data['Volume_Direction'] * data['TickVol']
    data['OBV'] = data['OBV'].cumsum()

    # Calculate Average Directional Index (ADX)
    adx_window = 96
    data['High_Low'] = data['High'] - data['Low']
    data['High_Prev_Close'] = abs(data['High'] - data['Close'].shift(1))
    data['Low_Prev_Close'] = abs(data['Low'] - data['Close'].shift(1))
    data['+DM'] = data[['High_Low', 'High_Prev_Close']].apply(lambda x: x['High_Low'] if x['High_Low'] > x['High_Prev_Close'] else 0, axis=1)
    data['-DM'] = data[['High_Low', 'Low_Prev_Close']].apply(lambda x: x['Low_Prev_Close'] if x['Low_Prev_Close'] > x['High_Low'] else 0, axis=1)
    data['+DI'] = (data['+DM'].rolling(window=adx_window).mean() / data['ATR'].rolling(window=adx_window).mean()) * 100
    data['-DI'] = (data['-DM'].rolling(window=adx_window).mean() / data['ATR'].rolling(window=adx_window).mean()) * 100
    data['DX'] = (abs(data['+DI'] - data['-DI']) / abs(data['+DI'] + data['-DI'])) * 100
    data['ADX'] = data['DX'].rolling(window=adx_window).mean()

    # Drop NaN values after adding indicators
    data.drop(columns=['High'], inplace=True)  # Drop Date column after creating Datetime index
    data.drop(columns=['Low'], inplace=True)
    data.drop(columns=['TickVol'], inplace=True)# Drop Date column after creating Datetime index
    data.dropna(inplace=True)
    total_observations = len(data)
    start_index = random.randint(0, total_observations - 96)
    data = data.iloc[start_index:total_observations]

    data['Close'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    class TradingEnvironment:
        def __init__(self, data):
            self.data = data
            self.current_step = 0
            self.fee_rate = 0.10
            self.lifespan = 0
            self.prev_eqr = 0
            self.positions = []  # Dictionary to track open buy and sell options
            self.portfolio = {
                'total_balance': 10000,  # Initial total balance (can be adjusted)
                'total_free_margin': 10000,  # Initial total free margin (can be adjusted)
                'total_equity': 10000,  # Initial total equity (can be adjusted)
                'total_buy_options': 0,  # Number of open buy options
                'total_sell_options': 0,  # Number of open sell options
                'sell_count': 0,  # Number of open sell options
                'buy_count': 0,  # Number of open sell options
                'start_hf': 0,
                'hold_fct': 0,
                'start_port': 0,
                'top_eqr': 0,
                'btm_eqr': 0,
                'check_bet': 0,
                'check_eqr': 0,
                'action_factor': 0,
                'start_risk_ratio': 0,
                'start_bet': 0,
                'hf': 0,
                'end_port': 0,
                'reward_port': 0,
                'rwp': 0,
                'holdings': 0,
                'pr': 0,
                'total_rw': 0, 
                'risk_penalty': 0,
                'end_bet': 0,
                'bet': 0,
                'eqr': 0,
                'erw': 0,
                'sc_per': 0,
                'bc_per': 0,
                'count_dist': 0,
                'trade_bias': 0,
                'pos_rw': 0,
                'reward': 0,
                'reward2': 0,
                'lifespan': 0
            }


        def reset(self):
            self.current_step = 0
            self.positions = []  # Reset positions dictionary
            self.portfolio = {
                'total_balance': 10000,  # Initial total balance (can be adjusted)
                'total_free_margin': 10000,  # Initial total free margin (can be adjusted)
                'total_equity': 10000,  # Initial total equity (can be adjusted)
                'total_buy_options': 0,  # Number of open buy options
                'total_sell_options': 0,  # Number of open sell options
                'sell_count': 0,  # Number of open sell options
                'buy_count': 0,  # Number of open sell options
                'start_hf': 0,
                'hold_fct': 0,
                'start_port': 0,
                'top_eqr': 0,
                'btm_eqr': 0,
                'check_bet': 0,
                'check_eqr': 0,
                'action_factor': 0,
                'start_risk_ratio': 0,
                'start_bet': 0,
                'hf': 0,
                'end_port': 0,
                'reward_port': 0,
                'rwp': 0,
                'holdings': 0,
                'pr': 0,
                'total_rw': 0, 
                'risk_penalty': 0,
                'end_bet': 0,
                'bet': 0,
                'eqr': 0,
                'erw': 0,
                'sc_per': 0,
                'bc_per': 0,
                'count_dist': 0,
                'trade_bias': 0,
                'pos_rw': 0,
                'reward': 0,
                'reward2': 0,
                'lifespan': 0
            }

        def get_state(self):
            current_data = self.data.iloc[self.current_step]  # Assuming 'data' is a pandas DataFrame
            current_features = np.array([
                current_data['Close'],
                current_data['MA_1'],
                current_data['MA_2'],
                current_data['MA_3'],
                current_data['MA_4'],
                current_data['MA_5'],
                current_data['MA_6'],
                current_data['RSI'],
                current_data['Short_MA'],
                current_data['Long_MA'],
                current_data['MACD'],
                current_data['Signal_Line'],
                current_data['Rolling_Mean'],
                current_data['Upper_Band'],
                current_data['Lower_Band'],
                current_data['Lowest_Low'],
                current_data['Highest_High'],
                current_data['%K'],
                current_data['%D'],
                current_data['ROC'],
                current_data['TR'],
                current_data['ATR'],
                current_data['Volume_Direction'],
                current_data['OBV'],
                current_data['+DM'],
                current_data['-DM'],
                current_data['+DI'],
                current_data['-DI'],
                current_data['DX'],
                current_data['ADX']
                # ... (add more features as needed)
            ], dtype=np.float32)

            # Portfolio information
            portfolio_state = np.array([
                self.portfolio['total_balance'],
                self.portfolio['total_free_margin'],
                self.portfolio['total_equity'],
                self.portfolio['total_buy_options'],
                self.portfolio['total_sell_options'],
                self.portfolio['sell_count'],
                self.portfolio['buy_count'],
                self.portfolio['start_hf'],
                self.portfolio['hold_fct'],
                self.portfolio['start_port'],
                self.portfolio['top_eqr'],
                self.portfolio['btm_eqr'],
                self.portfolio['check_bet'],
                self.portfolio['check_eqr'],
                self.portfolio['action_factor'],
                self.portfolio['start_risk_ratio'],
                self.portfolio['start_bet'],
                self.portfolio['hf'],
                self.portfolio['end_port'],
                self.portfolio['reward_port'],
                self.portfolio['rwp'],
                self.portfolio['holdings'],
                self.portfolio['pr'],
                self.portfolio['total_rw'],
                self.portfolio['risk_penalty'],
                self.portfolio['end_bet'],
                self.portfolio['bet'],
                self.portfolio['eqr'],
                self.portfolio['erw'],
                self.portfolio['sc_per'],
                self.portfolio['bc_per'],
                self.portfolio['count_dist'],
                self.portfolio['trade_bias'],
                self.portfolio['pos_rw'],
                self.portfolio['reward'],
                self.portfolio['reward2'],
                self.portfolio['lifespan']
            ], dtype=np.float32)

            # Concatenate current features and portfolio information
            state = np.concatenate((current_features, portfolio_state))

            return state
    
    
        def step(self, action, reward, reward2, agent_switch):
            print("-------------------------------------------------------------")
            print("-------------------------------------------------------------")
            print("-------------------------------------------------------------")
            print("-------------------------------------------------------------")
            original_action = action
            if os.path.exists(sc_filename):
              sell_count = load_sc(sc_filename)
            else:
              sell_count = 0
            print(f"Sell_count is {sell_count}")
            if os.path.exists(bc_filename):
              buy_count = load_bc(bc_filename)
            else:
              buy_count = 0
            print(f"Buy_count is {buy_count}")

            start_hf = 0
            max_holding_period = (96*5)-1
            holding_period_factors = []
            for position in self.positions:
                holding_period = position['open_step']
                holding_period_factor = 1 + (holding_period / max_holding_period)  # Adjust this as needed
                holding_period_factors.append(holding_period_factor)
            start_hf = sum(holding_period_factors)
            hold_fct = start_hf - len(self.positions)
            self.portfolio['start_hf'] = start_hf
            self.portfolio['hold_fct'] = hold_fct

            # Calculate total profit from buy positions
            total_buy_profit = 0
            for position in self.positions:
                if position['type'] == 'buy':
                    buy_profit = ((self.data['Close'][self.current_step] - position['open_price'])*10000) * (position['position_size'])
                    position['current_profit'] = buy_profit  # Update current profit for buy positions
                    fee = self.fee_rate * position['current_profit']
                    position['current_profit'] -= abs(fee)
                    total_buy_profit += position['current_profit']
 
            # Calculate total profit from sell positions
            total_sell_profit = 0
            for position in self.positions:
                if position['type'] == 'sell':
                    sell_profit = ((position['open_price'] - self.data['Close'][self.current_step])*10000) * (position['position_size'])
                    position['current_profit'] = sell_profit  # Update current profit for sell positions
                    fee = self.fee_rate * position['current_profit']
                    position['current_profit'] -= abs(fee)
                    total_sell_profit += position['current_profit']

            total_open_positions_profit = total_buy_profit + total_sell_profit
            start_port = total_open_positions_profit
            self.portfolio['start_port'] = start_port
            
            
            fee = 0
            prev_total_balance = self.portfolio['total_balance']
            pz = self.portfolio['total_free_margin']/1000
            if pz > 10:
              pz = 10
            action_factor = 1
            ATR = self.data['ATR'][self.current_step]
            if len(self.positions) == 0:
              bet_scale = pz * (self.portfolio['total_free_margin']*0.01)
            else:
              bet_scale = (self.portfolio['total_balance']-self.portfolio['total_free_margin'])
            scaled_value = ((ATR * 10000)*pz)/bet_scale
            if scaled_value < 0.20:
              scaled_value = 0.20
            print("ATR is ", scaled_value)
            top_eqr = (2)*scaled_value
            btm_eqr = (-1)*scaled_value
            self.portfolio['top_eqr'] = top_eqr
            self.portfolio['btm_eqr'] = btm_eqr

            print("Top Equity Target is: ", top_eqr)
            print("Bottom Equity Target is: ", btm_eqr)
            
            rsi = self.data['RSI'][self.current_step]
            obv = self.data['OBV'][self.current_step]
            short_ma = self.data['Short_MA'][self.current_step]
            long_ma = self.data['Long_MA'][self.current_step]
            
            tfmc = self.portfolio['total_free_margin']
            check_bet = (self.portfolio['total_balance']-tfmc)
            if start_port != 0 and check_bet != 0:
              check_eqr = start_port/check_bet
            else:
              check_eqr = 0
            eqrs = load_eqrs(eqrs_status_filename)
            if len(self.positions) >= 1:
                if check_eqr > top_eqr:
                  eqrs = 1
                  save_eqrs(eqrs, eqrs_status_filename)
            if len(self.positions) == 0:
                eqrs = 0
                save_eqrs(eqrs, eqrs_status_filename)
            print("the check eqr is ", check_eqr)
            print("the eqr status is ", eqrs)
            if action == 0 or action == 1:
              if action != 4:
                pzc = pz * (self.portfolio['total_free_margin']*.01)
                tfmc = self.portfolio['total_free_margin'] - pzc 
                rrc = (self.portfolio['total_balance']-tfmc) / self.portfolio['total_balance']
                rrc = round(rrc,2)
                check_bet = (self.portfolio['total_balance']-tfmc)
                if start_port != 0 and check_bet != 0:
                  check_eqr = start_port/check_bet
                else:
                  check_eqr = 0
                if len(self.positions)>=1:
                  if len(self.positions) == 1:
                    hdg_eqr = top_eqr
                  elif len(self.positions)==2:
                    hdg_eqr = top_eqr
                  elif len(self.positions)==3:
                    hdg_eqr = top_eqr
                  else:
                    hdg_eqr = top_eqr*1
                  print("Hedge Equity Target is: ", hdg_eqr)
                  if check_eqr < hdg_eqr:
                    print("Change Action 1")
                    action = 4
                    action_factor = -1
              if action != 4:
                pzc = pz * (self.portfolio['total_free_margin']*.01)
                tfmc = self.portfolio['total_free_margin'] - pzc 
                rrc = (self.portfolio['total_balance']-tfmc) / self.portfolio['total_balance']
                rrc = round(rrc,2)
                if rrc > .10:
                  if action == 0:
                    action = 4
                    action_factor = -1
                  if action == 1:
                    action = 4
                    action_factor = -1
            if action == 2 or action == 3 or action == 5:
              tfmc = self.portfolio['total_free_margin']
              check_bet = (self.portfolio['total_balance']-tfmc)
              if start_port != 0 and check_bet != 0:
                check_eqr = start_port/check_bet
              else:
                check_eqr = 0
              if btm_eqr < check_eqr < top_eqr:
                action = 4
                action_factor = -1
              if eqrs == 1:
                if check_eqr < self.prev_eqr:
                  action = 5
                  eqrs = 0
                  save_eqrs(eqrs, eqrs_status_filename)
            if action == 4:
              tfmc = self.portfolio['total_free_margin']
              check_bet = (self.portfolio['total_balance']-tfmc)
              if start_port != 0 and check_bet != 0:
                check_eqr = start_port/check_bet
              else:
                check_eqr = 0
              if check_eqr < btm_eqr:
                action = 5
                action_factor = -1
            if hold_fct >= 1:
                action = 5
                action_factor = -1
            if action == 0:
              buy_count += 1
            if action == 1:
              sell_count += 1
            #save_sell_count(sell_count, sc_filename)
            #save_buy_count(buy_count, bc_filename)
            self.portfolio['sell_count'] = sell_count
            self.portfolio['buy_count'] = buy_count
            self.portfolio['action_factor'] = action_factor
            self.portfolio['check_bet'] = check_bet
            self.portfolio['check_eqr'] = check_eqr
            if self.positions:
                for position in self.positions:
                    position['open_step'] += 1
            print("Original Action is:", original_action)
            print("Action is:", action)
            print("Current Price is:", self.data['Close'][self.current_step])

            self.portfolio['total_equity'] = self.portfolio['total_balance']
            for position in self.positions:
                pt = position['current_profit']
                self.portfolio['total_equity'] += pt
            # Calculate reward based on the action and current state
            #reward = 0  # You need to define how to calculate the reward based on the action and state
            start_risk_ratio = round((self.portfolio['total_balance']-self.portfolio['total_free_margin']) / self.portfolio['total_balance'],2)
            start_bet = self.portfolio['total_balance'] - self.portfolio['total_free_margin']
            self.portfolio['start_risk_ratio'] = start_risk_ratio
            self.portfolio['start_bet'] = start_bet
            print("-------------------------------------------------------------")
            print("Portfolio at start of step:")
            print("Total Balance:", self.portfolio['total_balance'])
            print("Total Free Margin:", self.portfolio['total_free_margin'])
            print("Total Risk Margin:", start_bet)
            print("Total Equity:", self.portfolio['total_equity'])
            print("Total Buy Options:", self.portfolio['total_buy_options'])
            print("Total Profit from Open Buy Options:", total_buy_profit)
            print("Total Sell Options:", self.portfolio['total_sell_options'])
            print("Total Profit from Open Sell Options:", total_sell_profit)
            print("Total Profit from All Open Options:", total_open_positions_profit)
            
            # Update positions and portfolio based on the action
            if action == 0:  # Open a buy option
                position_size = pz
                self.positions.append({
                    'type': 'buy',
                    'position_size': position_size,
                    'open_price': self.data['Close'][self.current_step],
                    'current_profit': 0,
                    'life_count': 0,
                    'total_margin': position_size * (self.portfolio['total_free_margin']*.01),
                    'open_step': 1
                })
                self.portfolio['total_buy_options'] += 1
                self.portfolio['total_free_margin'] -= position_size * (self.portfolio['total_free_margin']*.01)
                
            elif action == 1:  # Open a sell option
                position_size = pz
                self.positions.append({
                    'type': 'sell',
                    'position_size': position_size,
                    'open_price': self.data['Close'][self.current_step],
                    'current_profit': 0,
                    'life_count': 0,
                    'total_margin': position_size * (self.portfolio['total_free_margin']*.01),
                    'open_step': 1
                })
                self.portfolio['total_sell_options'] += 1
                self.portfolio['total_free_margin'] -= position_size * (self.portfolio['total_free_margin']*.01)
                
            elif action == 2:  # Close the first open buy option
                for position in self.positions:
                    if position['type'] == 'buy':
                        buy_profit = position['current_profit']
                        self.portfolio['total_free_margin'] += position['total_margin']
                        self.portfolio['total_free_margin'] += buy_profit
                        self.portfolio['total_balance'] += buy_profit
                        self.positions.remove(position)  # Remove the buy position from self.positions
                        self.portfolio['total_buy_options'] -= 1
                        break
            
            elif action == 3:  # Close the first open sell option
                for position in self.positions:
                    if position['type'] == 'sell':
                        sell_profit = position['current_profit']
                        self.portfolio['total_free_margin'] += position['total_margin']
                        self.portfolio['total_free_margin'] += sell_profit
                        self.portfolio['total_balance'] += sell_profit
                        self.positions.remove(position)  # Remove the sell position from self.positions
                        self.portfolio['total_sell_options'] -= 1
                        break
            
            elif action == 4:  # Hold position (do nothing)
                for position in self.positions:
                    if position['type'] == 'buy':
                         position['current_profit'] = ((self.data['Close'][self.current_step] - position['open_price'])*10000) * (position['position_size'])
                    elif position['type'] == 'sell':
                         position['current_profit'] = ((position['open_price'] - self.data['Close'][self.current_step])*10000) * (position['position_size'])

            elif action == 5:  # Close all open buy and sell options
                positions_to_remove = []
                for position in self.positions:
                    if position['type'] == 'buy':
                        buy_profit = position['current_profit']
                        self.portfolio['total_balance'] += buy_profit
                        self.portfolio['total_free_margin'] += buy_profit
                        self.portfolio['total_free_margin'] += position['total_margin']
                        #self.positions.remove(position)
                        self.portfolio['total_buy_options'] -= 1
                    elif position['type'] == 'sell':
                        sell_profit = position['current_profit']
                        self.portfolio['total_balance'] += sell_profit
                        self.portfolio['total_free_margin'] += sell_profit
                        self.portfolio['total_free_margin'] += position['total_margin']
                        #self.positions.remove(position)
                        self.portfolio['total_sell_options'] -= 1
                
                    positions_to_remove.append(position)
                
                for position in positions_to_remove:
                    self.positions.remove(position)    
        
            self.portfolio['total_equity'] = self.portfolio['total_balance']
            for position in self.positions:
                pt = position['current_profit']
                self.portfolio['total_equity'] += pt
            print("-------------------------------------------------------------")
            #positioning life span 
            if len(self.positions) == 0:
              if action == 2 or action == 3 or action == 5:
                self.lifespan += 1
              else:
                self.lifespan = 0
            if len(self.positions) >= 1:
              if action == 0 or action == 1:
                self.lifespan = 0
              else:
                self.lifespan += 1
            self.portfolio['lifespan'] = self.lifespan
            print("The Lifespan of the positioning is: ", self.lifespan)
            print("-------------------------------------------------------------")
            print("Portfolio Rewards/Penalty System:")
            hf = 0
            max_holding_period = (96*5)-1
            holding_period_factors = []
            for position in self.positions:
                holding_period = position['open_step']
                holding_period_factor = 1 + (holding_period / max_holding_period)  # Adjust this as needed
                holding_period_factors.append(holding_period_factor)
            hf = sum(holding_period_factors)
            self.portfolio['hf'] = hf
            # Action Rewards ----
            total_buy_profit = 0
            for position in self.positions:
                if position['type'] == 'buy':
                    buy_profit = ((self.data['Close'][self.current_step] - position['open_price'])*10000) * (position['position_size'])
                    position['current_profit'] = buy_profit  # Update current profit for buy positions
                    fee = self.fee_rate * position['current_profit']
                    position['current_profit'] -= abs(fee)
                    total_buy_profit += position['current_profit']
            total_sell_profit = 0
            for position in self.positions:
                if position['type'] == 'sell':
                    sell_profit = ((position['open_price'] - self.data['Close'][self.current_step])*10000) * (position['position_size'])
                    position['current_profit'] = sell_profit  # Update current profit for sell positions
                    fee = self.fee_rate * position['current_profit']
                    position['current_profit'] -= abs(fee)
                    total_sell_profit += position['current_profit']
            total_open_positions_profit = total_buy_profit + total_sell_profit
            end_port = total_open_positions_profit
            reward_port = end_port + start_port
            self.portfolio['end_port'] = end_port
            self.portfolio['reward_port'] = reward_port
            rwp = 0
            if reward_port > 0:
              if agent_switch == 1:
                if len(self.positions)==0 and action == 4:
                  reward += 0
                  rwp += 0
                else:
                  reward += 1
                  rwp += 1
              else:
                if len(self.positions)==0 and action == 4:
                  reward2 += 0
                  rwp += 0
                else:
                  reward2 += 1
                  rwp += 1
            elif reward_port < 0:
              if agent_switch == 1:
                if len(self.positions)==0 and action == 4:
                   reward += 0
                   rwp += 0
                else:
                  reward += -1
                  rwp += -1
              else:
                if len(self.positions)==0 and action == 4:
                   reward2 += 0
                   rwp += 0
                else:
                  reward2 += -1
                  rwp += -1
            else:
              reward += 0
              reward2 += 0
            print(f"Total Equity Reward/Penalty: {rwp}")
            self.portfolio['rwp'] = rwp
            # Portfolio Rewards ----
            holdings = 0
            hdr = (self.portfolio['total_balance']/10000)
            if hdr < 1:
              hdr = hdr*-1
            if self.portfolio['total_balance'] < 10000:
                if agent_switch == 1:
                  if len(self.positions)==0 and action == 4:
                     holdings += 0
                     reward += 0
                  else:
                    holdings += hdr
                    reward += hdr
                else:
                  if len(self.positions)==0 and action == 4:
                     holdings += 0
                     reward2 += 0
                  else:
                    holdings += hdr
                    reward2 += hdr
            elif self.portfolio['total_balance'] >= 10000:
                if agent_switch == 1:
                  if len(self.positions)==0 and action == 4:
                     holdings += 0
                     reward += 0
                  else:
                    holdings += hdr
                    reward += hdr
                else:
                  if len(self.positions)==0 and action == 4:
                     holdings += 0
                     reward2 += 0
                  else:
                    holdings += hdr
                    reward2 += hdr
            else:
              reward += 0
              reward2 += 0
            print("Current Portfolio Balance Reward/Penalty:", holdings)
            self.portfolio['holdings'] = holdings
            # Store the new total balance after updating positions and portfolio
            new_total_balance = self.portfolio['total_balance']
            # Calculate reward based on the change in total balance
            portfolio_reward = new_total_balance - prev_total_balance
            # Update the reward based on the change in total balance
            pr = 0
            if portfolio_reward > 0:
              if agent_switch == 1:
                if len(self.positions)==0 and action == 4:
                   pr += 0
                   reward += 0
                else:
                   pr += self.lifespan 
                   reward += self.lifespan
              else:
                if len(self.positions)==0 and action == 4:
                   pr += 0
                   reward2 += 0
                else:
                  pr += self.lifespan
                  reward2 += self.lifespan
            elif portfolio_reward < 0:
              if agent_switch == 1:
                if len(self.positions)==0 and action == 4:
                   pr += 0
                   reward += 0
                else:
                  pr += self.lifespan *-1
                  reward += self.lifespan *-1
              else:
                if len(self.positions)==0 and action == 4:
                   pr += 0
                   reward2 += 0
                else:
                  pr += self.lifespan *-1
                  reward2 += self.lifespan *-1
            else:
              reward += 0
              reward2 += 0
            print("Closing Portfolio Balance Reward/Penalty:", pr)
            self.portfolio['pr'] = pr
            afrw = 0
            if action_factor == 1:
              if agent_switch == 1:
                if len(self.positions)==0 and action == 4:
                   reward += 0
                   afrw +=0
                else:
                  reward += 1
                  afrw += 1
              else:
                if len(self.positions)==0 and action == 4:
                   reward2 += 0
                   afrw += 0
                else:
                  reward2 += 1
                  afrw += 1
            else:
              if agent_switch == 1:
                if len(self.positions)==0 and action == 4:
                   reward += 0
                   afrw += 0
                else:
                  reward += -1
                  afrw += -1
              else:
                if len(self.positions)==0 and action == 4:
                   reward += 0
                   afrw += 0
                else:
                  afrw += -1
                  reward2 += -1
            print("Action Correction Reward/Penalty", afrw)
            total_rw = pr + holdings + rwp + afrw
            print("--------------------------------------------------------")
            # Penalty if balance is below minimum threshold
            if action == 0 or action == 1 or action == 4:
                risk_ratio = round((self.portfolio['total_balance']-self.portfolio['total_free_margin']) / self.portfolio['total_balance'],2)
                rr_ratio = round(start_risk_ratio + risk_ratio,2)
                risk_threshold = 0.10
                # Check if risk exceeds the threshold
                risk_penalty = 0
                if risk_ratio > risk_threshold:
                    # Calculate penalty based on the exceeding risk
                    risk_penalty += -1
                elif 0 < risk_ratio <= risk_threshold:
                    # Calculate penalty based on the exceeding risk
                    risk_penalty += 1
            else:
                risk_ratio = round((self.portfolio['total_balance']-self.portfolio['total_free_margin']) / self.portfolio['total_balance'],2)
                rr_ratio = round(start_risk_ratio + risk_ratio,2)
                risk_threshold = 0.10
                # Check if risk exceeds the threshold
                risk_penalty = 0
                if start_risk_ratio > risk_threshold:
                    # Calculate penalty based on the exceeding risk
                    risk_penalty += -1
                elif 0 < start_risk_ratio <= risk_threshold:
                    # Calculate penalty based on the exceeding risk
                    risk_penalty += 1
            self.portfolio['risk_penalty'] = risk_penalty
            end_bet = self.portfolio['total_balance'] - self.portfolio['total_free_margin']
            bet = start_bet + end_bet
            self.portfolio['end_bet'] = end_bet
            self.portfolio['bet'] = bet
            if reward_port != 0 and bet != 0:
              eqr = reward_port/bet
            else:
              eqr = 0            
            erw = 0
            if eqr >= 0:
                erw += 1
            elif eqr < 0:
                erw += -1
            self.portfolio['eqr'] = eqr
            self.portfolio['erw'] = erw
                
            total_actions = sell_count + buy_count
            if sell_count != 0:
              sc_per = sell_count/total_actions
            else:
              sc_per = 0
            if buy_count != 0:
              bc_per = buy_count/total_actions
            else:
              bc_per = 0
            # Set a range for the comparison
            range_threshold = 0.1
            # Check if the absolute difference between bc_per and sc_per is within the range_threshold
            if abs(bc_per - sc_per) <= range_threshold:
               count_dist = 1
            else:
              if self.portfolio['total_buy_options'] >= 1 and sc_per > bc_per:
                 count_dist = 1
              elif self.portfolio['total_sell_options'] >= 1 and bc_per > sc_per:
                count_dist = 1
              else:
                if reward_port > 0:
                  count_dist = 1
                else:
                  count_dist = -1
            self.portfolio['sc_per'] = sc_per
            self.portfolio['bc_per'] = bc_per
            self.portfolio['count_dist'] = count_dist
            self.portfolio['trade_bias'] = abs(bc_per - sc_per)
            print("Trade Bias is: ", abs(bc_per - sc_per))
            print("Sell Distribution is: ", sc_per)
            print("Buy Distribution is: ", bc_per)
            print("Trade Type Distribution Status is: ", count_dist)
            print("Action Factor is: ", action_factor)
            if risk_penalty >= 0 and erw == 1 and count_dist == 1:
              pos_rw = 1
            else:
              pos_rw = -1
            self.portfolio['pos_rw'] = pos_rw
            print("Equity Ratio is: ", eqr)
            print("Equity Ratio Factor: ", erw)
            if action == 0 or action == 1 or action == 4:
              print(f"Risk Ratio is: {risk_ratio}")
            else:
              print(f"Risk Ratio is: {start_risk_ratio}")
            print(f"Risk Ratio Factor: {risk_penalty}")
            print(f"Overall Equity Risk Ratio Factor: {pos_rw}")
            print("--------------------------------------------------------")
            if hf != 0:
                print("Total Step Holding/Period Factor:", hf)
                if len(self.positions)==0 and action == 4:
                  rwf = 0
                  total_rw += 0
                  if agent_switch == 1:
                    reward += rwf
                  else:
                    reward2 += rwf
                else:
                  rwf = (pos_rw*hf)+eqr
                  total_rw += rwf
                  if agent_switch == 1:
                    reward += rwf
                  else:
                    reward2 += rwf
            else:
                if start_hf != 0:
                  print("Total Step Holding/Period Factor:", start_hf)
                  if len(self.positions)==0 and action == 4:
                    rwf = 0
                    total_rw += 0
                    if agent_switch == 1:
                      reward += rwf
                    else:
                      reward2 += rwf
                  else:
                    rwf = (pos_rw*start_hf)+eqr
                    total_rw += rwf
                    if agent_switch == 1:
                      reward += rwf
                    else:
                      reward2 += rwf
                else:
                  if len(self.positions)==0 and action == 4:
                    total_rw += 0
                    if agent_switch == 1:
                      reward += 0
                    else: 
                      reward2 += 0
                  else:
                    eqrw = eqr
                    if eqr > 0:
                      eqrw = (pos_rw)*eqr
                    total_rw += eqrw
                    if agent_switch == 1:
                      reward += eqrw
                    else: 
                      reward2 += eqrw
                      
            if len(self.positions)==0 and action == 4:
              total_rw += -0.01
              if agent_switch == 1:
                 reward += -0.01
              else: 
                 reward2 += -0.01
            
            fault = False
            if fault == False:
              if agent_switch == 1:
                if len(self.positions)==0 and action == 4:
                  reward += 0
                else:
                  reward += 1
              else:
                if len(self.positions)==0 and action == 4:
                  reward2 += 0
                else:
                  reward2 += 1
            self.portfolio['total_rw'] = total_rw
            self.portfolio['reward'] = reward
            self.portfolio['reward2'] = reward2

            print(f"Total Step Rewards: {total_rw}")
            if agent_switch == 1:
              print(f"Total Rewards: {reward}")
            else:
              print(f"Total Rewards2: {reward2}")
            print("The Overall Reward is ", reward+reward2)
            # Calculate the next state of the environment
            next_state = self.get_state()
            print("-------------------------------------------------------------")
            print("Portfolio after step:")
            print("Total Balance:", self.portfolio['total_balance'])
            print("Total Free Margin:", self.portfolio['total_free_margin'])
            print("Total Equity:", self.portfolio['total_equity'])
            print("Total Buy Options:", self.portfolio['total_buy_options'])
            print("Total Profit from Open Buy Options:", total_buy_profit)
            print("Total Sell Options:", self.portfolio['total_sell_options'])
            print("Total Profit from Open Sell Options:", total_sell_profit)
            print("Total Profit from All Open Options:", total_open_positions_profit)
        
            total_bal = self.portfolio['total_balance']
            total_eq = self.portfolio['total_equity']
            boc = self.portfolio['total_buy_options']
            soc = self.portfolio['total_sell_options']
            
            # Update current step
            self.current_step += 1
            self.prev_eqr = eqr
            # Determine if the episode is done (you need to define your own termination conditions)
            done = True  # Modify this based on your termination conditions
            return next_state, reward, reward2, total_bal, total_eq, boc, soc, total_buy_profit, total_sell_profit, total_open_positions_profit, risk_ratio, self.positions, original_action, action_factor, done
    
    # DQN Agent
    class DQNAgent:
        def __init__(self, state_size, action_size, env, model_filename=None, model_filename2=None):
            self.state_size = state_size
            self.action_size = action_size
            self.memory = deque(maxlen=max_mem)
            self.memory2 = deque(maxlen=max_mem)
            self.gamma = 0.95  # Discount factor
            self.epsilon = epi  # Exploration rate
            self.epsilon2 = epi2  # Exploration rate
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.learning_rate = 0.001
            self.model = self._build_model()
            self.model2 = self._build_model()
            self.env = env  # Store the environment instance
            self.model_filename = model_filename
            self.model_filename2 = model_filename2

            # Load pre-trained model if it exists
            if model_filename:
                try:
                    self.model.load_weights(model_filename)
                    print("Loaded model weights from", model_filename)
                except:
                    print("Could not load model weights. Starting with random weights.")
            # Load pre-trained model if it exists
            if model_filename2:
                try:
                    self.model2.load_weights(model_filename2)
                    print("Loaded model weights from", model_filename2)
                except:
                    print("Could not load model weights. Starting with random weights.")

        def _build_model(self):
           input_layer = Input(shape=(self.state_size,))
           shared_layer1 = Dense(512, activation='relu')(input_layer)
           shared_layer2 = Dense(1024, activation='relu')(shared_layer1)
           reshaped_state = Reshape((1, 1024))(shared_layer2)
           # Transformer Layer
           num_transformer_layers = 3
           transformer_output = reshaped_state
           for _ in range(num_transformer_layers):
             transformer_output = MultiHeadAttention(num_heads=32, key_dim=self.state_size // 64)(transformer_output, transformer_output)
             transformer_output = LayerNormalization()(transformer_output)
             transformer_output = Dropout(0.1)(transformer_output)
           transformer_output = Flatten()(transformer_output)
           
           num_transformer_layers = 6
           transformer_output2 = reshaped_state
           for _ in range(num_transformer_layers):
             transformer_output2 = MultiHeadAttention(num_heads=32, key_dim=self.state_size // 64)(transformer_output2, transformer_output2)
             transformer_output2 = LayerNormalization()(transformer_output2)
             transformer_output2 = Dropout(0.1)(transformer_output2)
           transformer_output2 = Flatten()(transformer_output2)
             
           num_transformer_layers = 9
           transformer_output3 = reshaped_state
           for _ in range(num_transformer_layers):
             transformer_output3 = MultiHeadAttention(num_heads=32, key_dim=self.state_size // 64)(transformer_output3, transformer_output3)
             transformer_output3 = LayerNormalization()(transformer_output3)
             transformer_output3 = Dropout(0.1)(transformer_output3)
           transformer_output3 = Flatten()(transformer_output3)
             
           num_transformer_layers = 3
           transformer_output4 = reshaped_state
           for _ in range(num_transformer_layers):
             transformer_output4 = MultiHeadAttention(num_heads=32, key_dim=self.state_size // 64)(transformer_output4, transformer_output4)
             transformer_output4 = LayerNormalization()(transformer_output4)
             transformer_output4 = Dropout(0.1)(transformer_output4)
           transformer_output4 = Flatten()(transformer_output4)
             
           num_transformer_layers = 6
           transformer_output5 = reshaped_state
           for _ in range(num_transformer_layers):
             transformer_output5 = MultiHeadAttention(num_heads=32, key_dim=self.state_size // 64)(transformer_output5, transformer_output5)
             transformer_output5 = LayerNormalization()(transformer_output5)
             transformer_output5 = Dropout(0.1)(transformer_output5)
           transformer_output5 = Flatten()(transformer_output5)
             
           num_transformer_layers = 9
           transformer_output6 = reshaped_state
           for _ in range(num_transformer_layers):
             transformer_output6 = MultiHeadAttention(num_heads=32, key_dim=self.state_size // 64)(transformer_output6, transformer_output6)
             transformer_output6 = LayerNormalization()(transformer_output6)
             transformer_output6 = Dropout(0.1)(transformer_output6)
           transformer_output6 = Flatten()(transformer_output6)

           # Value Streams
           value_stream1 = Dense(1024, activation='relu')(transformer_output)
           value_stream1 = LayerNormalization()(value_stream1)
           value1 = Dense(1, activation='linear')(value_stream1)

           value_stream2 = Dense(1024, activation='relu')(transformer_output2)
           value_stream2 = LayerNormalization()(value_stream2)
           value2 = Dense(1, activation='linear')(value_stream2)
           
           value_stream3 = Dense(1024, activation='relu')(transformer_output3)
           value_stream3 = LayerNormalization()(value_stream3)
           value3 = Dense(1, activation='linear')(value_stream3)
           
           value_stream4 = Dense(1024, activation='relu')(transformer_output4)
           value_stream4 = LayerNormalization()(value_stream4)
           value4 = Dense(1, activation='linear')(value_stream4)
           
           value_stream5 = Dense(1024, activation='relu')(transformer_output5)
           value_stream5 = LayerNormalization()(value_stream5)
           value5 = Dense(1, activation='linear')(value_stream5)
           
           value_stream6 = Dense(1024, activation='relu')(transformer_output6)
           value_stream6 = LayerNormalization()(value_stream6)
           value6 = Dense(1, activation='linear')(value_stream6)

           # Advantage Streams
           advantage_stream1 = Dense(1024, activation='relu')(transformer_output)
           advantage_stream1 = LayerNormalization()(advantage_stream1)
           advantage1 = Dense(self.action_size, activation='linear')(advantage_stream1)

           advantage_stream2 = Dense(1024, activation='relu')(transformer_output2)
           advantage_stream2 = LayerNormalization()(advantage_stream2)
           advantage2 = Dense(self.action_size, activation='linear')(advantage_stream2)
           
           advantage_stream3 = Dense(1024, activation='relu')(transformer_output3)
           advantage_stream3 = LayerNormalization()(advantage_stream3)
           advantage3 = Dense(self.action_size, activation='linear')(advantage_stream3)
           
           advantage_stream4 = Dense(1024, activation='relu')(transformer_output4)
           advantage_stream4 = LayerNormalization()(advantage_stream4)
           advantage4 = Dense(self.action_size, activation='linear')(advantage_stream4)
           
           advantage_stream5 = Dense(1024, activation='relu')(transformer_output5)
           advantage_stream5 = LayerNormalization()(advantage_stream5)
           advantage5 = Dense(self.action_size, activation='linear')(advantage_stream5)
           
           advantage_stream6 = Dense(1024, activation='relu')(transformer_output6)
           advantage_stream6 = LayerNormalization()(advantage_stream6)
           advantage6 = Dense(self.action_size, activation='linear')(advantage_stream6)

           # Compute Q-values using value and advantage streams
           q_values1 = value1 + (advantage1 - K.mean(advantage1, axis=1, keepdims=True))
           q_values2 = value2 + (advantage2 - K.mean(advantage2, axis=1, keepdims=True))
           q_values3 = value3 + (advantage3 - K.mean(advantage3, axis=1, keepdims=True))
           q_values4 = value4 + (advantage4 - K.mean(advantage4, axis=1, keepdims=True))
           q_values5 = value5 + (advantage5 - K.mean(advantage5, axis=1, keepdims=True))
           q_values6 = value6 + (advantage6 - K.mean(advantage6, axis=1, keepdims=True))

           # Combine Q-values into a single output
           q_values = (q_values1 + q_values2 + q_values3 + q_values4 + q_values5 + q_values6)/6

           model = Model(inputs=input_layer, outputs=q_values)
           model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
           return model


        def remember(self, state, action, reward, next_state, done):
            if agent_switch == 1:
              self.memory.append((state, action, reward, next_state, done))
            else:
              self.memory2.append((state, action, reward, next_state, done))

        def act(self, state):
            if agent_switch == 1:
              # Exploration: choose a random action if epsilon-greedy exploration
              if np.random.rand() <= self.epsilon:
                # Get a list of suitable actions based on open positions
                suitable_actions = []
            
                if not self.env.positions:# and trend_direction == 0:  # If there are no open positions
                    suitable_actions.extend([0,1,4])  # Choose from actions 0, 1, or 4
                else:
                    if any(position['type'] == 'buy' for position in self.env.positions):# and trend_direction == 0:
                        suitable_actions.extend([2, 4, 5])  # Add actions 2 and 5 if there are open buy positions
                    if any(position['type'] == 'sell' for position in self.env.positions):# and trend_direction == 0:
                        suitable_actions.extend([3, 4, 5])  # Add actions 3 and 5 if there are open sell positions             
                    if any(position['type'] == 'buy' for position in self.env.positions) and any(position['type'] == 'sell' for position in self.env.positions):# and trend_direction == 0:
                        suitable_actions.extend([0, 1, 2, 3, 4, 5])
                return np.random.choice(suitable_actions)
            else:
              # Exploration: choose a random action if epsilon-greedy exploration
              if np.random.rand() <= self.epsilon2:
                # Get a list of suitable actions based on open positions
                suitable_actions = []
            
                if not self.env.positions:# and trend_direction == 0:  # If there are no open positions
                    suitable_actions.extend([0,1,4])  # Choose from actions 0, 1, or 4
                else:
                    if any(position['type'] == 'buy' for position in self.env.positions):# and trend_direction == 0:
                        suitable_actions.extend([2, 4, 5])  # Add actions 2 and 5 if there are open buy positions
                    if any(position['type'] == 'sell' for position in self.env.positions):# and trend_direction == 0:
                        suitable_actions.extend([3, 4, 5])  # Add actions 3 and 5 if there are open sell positions             
                    if any(position['type'] == 'buy' for position in self.env.positions) and any(position['type'] == 'sell' for position in self.env.positions):# and trend_direction == 0:
                        suitable_actions.extend([0, 1, 2, 3, 4, 5])
                return np.random.choice(suitable_actions)
    
            # Exploitation: choose the action with the highest predicted reward
            q_values = self.model.predict(state)
            q_values2 = self.model2.predict(state)
            q_values = (q_values+q_values2)/2
            # Get suitable actions based on open positions
            suitable_actions = []
            if not self.env.positions:# and trend_direction == 0:  # If there are no open positions
                suitable_actions.extend([0,1,4])  # Choose from actions 0, 1, or 4
            else:
                if any(position['type'] == 'buy' for position in self.env.positions):# and trend_direction == 0:
                    suitable_actions.extend([2, 4, 5])  # Add actions 2 and 5 if there are open buy positions
                if any(position['type'] == 'sell' for position in self.env.positions):# and trend_direction == 0:
                    suitable_actions.extend([3, 4, 5])  # Add actions 3 and 5 if there are open sell positions
                if any(position['type'] == 'buy' for position in self.env.positions) and any(position['type'] == 'sell' for position in self.env.positions):# and trend_direction == 0:
                    suitable_actions.extend([0, 1, 2, 3, 4, 5])
        
            # Filter q_values to keep only suitable actions
            filtered_q_values = [q_values[0][action] for action in suitable_actions]
            return suitable_actions[np.argmax(filtered_q_values)]

    
    model_filename = 'agent1.h5'  # Define the model file name for saving and loading your trained model
    model_filename2 = 'agent2.h5'  # Define the model file name for saving and loading your trained model

    env = TradingEnvironment(data)  # Initialize your custom trading environment with appropriate data
    state_size = len(env.get_state())  # Adjust state size based on your environment's state representation
    action_size = 6  # Set the number of actions according to your trading actions (buy, sell, hold, etc.)
    dqn_agent = DQNAgent(state_size, action_size, env, model_filename, model_filename2)  # Initialize your DQN agent with the appropriate state and action sizes
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    csv_filename = f"results/test/test_{timestamp}.csv"
    episode_rewards_df = pd.DataFrame(columns=['Episode', 'Reward', 'Balance'])
    
    total_episodes = len(data)  # Set the total number of episodes based on your dataset
    batch_size = 96  # Adjust the batch size for experience replay
    batch_size2 = 96*5
    episode_rewards = []
    episode_count = 0
    episode_reward = 0
    episode_reward2 = 0
    episode_balance = 0
    episode_balances = []
    epi = epi
    epi2 = epi2
    # Load the pre-trained model if it exists
    if os.path.exists(model_filename):
        dqn_agent.model = load_model(model_filename)
        print(f"Loaded model from {model_filename}")
        # Load the pre-trained model if it exists
    if os.path.exists(model_filename2):
        dqn_agent.model2 = load_model(model_filename2)
        print(f"Loaded model from {model_filename2}")

    while episode_count < total_episodes:
        done = False
        failure_done = False
        while not done:
            if env.portfolio['total_equity'] < (10000*0.90) or env.portfolio['total_balance'] < (10000* 0.90):  # Define a custom method in your environment to determine if the episode is done
                done = True
                failure_done = True
                env.reset() 
                episode_reward = 0
                episode_reward2 = 0
                episode_balance = 0
                episode_count = 0
                episode_count += 1
                episode_rewards = []
                episode_balances = []
                # Handle episode termination logic if needed
                if reward >= reward2:
                  agent_switch = 1
                else:
                  agent_switch = 2
                print("Active agent is, ", agent_switch)
                print(f"Episode {episode_count} Terminated. Balance is Below Threshold.")
            else:
                state = env.get_state()
                state = np.reshape(state, [1, state_size])
                action = dqn_agent.act(state)
                next_state, reward, reward2, total_bal, total_eq, boc, soc, total_buy_profit, total_sell_profit, total_open_positions_profit, risk_ratio, ind_positions, original_action, action_factor, done = env.step(action, reward, reward2, agent_switch)  # Implement the step method in your environment
                next_state = np.reshape(next_state, [1, state_size])
                episode_balance = total_bal
                episode_reward += reward
                episode_reward2 += reward2
                episode_count += 1
                
                
                #if agent_switch == 1:
                #  if action_factor == -1:
                #    dqn_agent.remember(state, original_action, reward, next_state, done)
                #  else:
                #    dqn_agent.remember(state, action, reward, next_state, done)
                print(f"Step Reward: {reward}, Cumulative Reward: {episode_reward}")
                #else:
                #  if action_factor == -1:
                #    dqn_agent.remember(state, original_action, reward2, next_state, done)
                #  else:
                #    dqn_agent.remember(state, action, reward2, next_state, done)
                print(f"Step Reward: {reward2}, Cumulative Reward: {episode_reward2}")
                print(f"Episode: {episode_count} of Total Episodes: {total_episodes}")

        print(f"The Epsilon is {dqn_agent.epsilon}.")
        print(f"The Epsilon2 is {dqn_agent.epsilon2}.")
        # Visualize episode rewards
        episode_rewards.append(episode_reward+episode_reward2)
        episode_balances.append(episode_balance)
        if int(episode_count) != int(total_episodes):
          ind_pos_df = pd.DataFrame(ind_positions)
          ind_pos_df.to_csv('results/test/ind_positions.csv', index=False)
        episode_rewards_df = episode_rewards_df.append({'Episode': episode_count, 'Reward': reward+reward2, 'Balance': env.portfolio['total_balance'],
                                                        'total_equity': total_eq, 'total_buy_options': boc, 'total_sell_options': soc,
                                                        'total_op_profit': total_open_positions_profit, 'risk_ratio': risk_ratio}, 
                                                        ignore_index=True)
        episode_rewards_df.to_csv(csv_filename, index=False)
        #plt.figure(figsize=(12, 5))
        #plt.subplot(1, 2, 1)
        #plt.plot(episode_rewards)
        #plt.xlabel('Episode')
        #plt.ylabel('Total Reward')
        #plt.title('Agent Performance')
    
        #plt.subplot(1, 2, 2)
        #plt.plot(episode_balances, color='orange')
        #plt.xlabel('Episode')
        #plt.ylabel('Total Balance')
        #plt.title('Portfolio Performance')

        #plt.tight_layout()
        #plt.show()
    
        # Save the trained model9+78652
        #dqn_agent.model.save(model_filename)
        if agent_switch == 1:
          print(f"Trained model tested as {model_filename}")
        else:
          print(f"Trained model tested as {model_filename2}")

        # Experience replay
        #if len(dqn_agent.memory) > batch_size:
        #    dqn_agent.replay(batch_size)
        if action == 2 or action == 3 or action == 5:
          if len(env.positions) < 1:
            if reward >= reward2:
              agent_switch = 1
            else:
              agent_switch = 2
        if int(episode_count) == int(total_episodes):
          if env.portfolio['total_balance'] <= int((10000+(10000*.10))):
                failure_done = True
                env.reset() 
        if env.portfolio['total_balance'] >= int((10000+(10000*.10))):
            train = False
        if failure_done == True:
            done = True
            break
    return train, done  
            
data = pd.read_csv(r'data\data_2023.csv')
train = True
epi_filename = 'epi_value.pkl'
rw_filename = 'rw_value.pkl'
epi_filename2 = 'epi_value2.pkl'
rw_filename2 = 'rw_value2.pkl'
sc_filename = 'sc_value.pkl'
bc_filename = 'bc_value.pkl'
reset1_filename = 'reset1_value.pkl'
reset2_filename = 'reset2_value.pkl'
eqrs_status_filename = 'eqr_status_value.pkl'

import pickle

def save_sell_count(sell_count, filename):
    with open(filename, 'wb') as file:
        pickle.dump(sell_count, file)
# Load epi from a file if it exists
def load_sc(filename):
    try:
        with open(filename, 'rb') as file:
            sell_count = pickle.load(file)
            return sell_count
    except FileNotFoundError:
        return None
def save_buy_count(buy_count, filename):
    with open(filename, 'wb') as file:
        pickle.dump(buy_count, file)
# Load epi from a file if it exists
def load_bc(filename):
    try:
        with open(filename, 'rb') as file:
            buy_count = pickle.load(file)
            return buy_count
    except FileNotFoundError:
        return None
# Save epi to a file
def save_epi(epi, filename):
    with open(filename, 'wb') as file:
        pickle.dump(epi, file)

# Load epi from a file if it exists
def load_epi(filename):
    try:
        with open(filename, 'rb') as file:
            epi = pickle.load(file)
            return epi
    except FileNotFoundError:
        return None
# Save epi to a file
def save_epi2(epi2, filename):
    with open(filename, 'wb') as file:
        pickle.dump(epi2, file)

# Load epi from a file if it exists
def load_epi2(filename):
    try:
        with open(filename, 'rb') as file:
            epi2 = pickle.load(file)
            return epi2
    except FileNotFoundError:
        return None
# Save epi to a file
def save_rw(reward, filename):
    with open(filename, 'wb') as file:
        pickle.dump(reward, file)

# Load epi from a file if it exists
def load_rw(filename):
    try:
        with open(filename, 'rb') as file:
            reward = pickle.load(file)
            return reward
    except FileNotFoundError:
        return None
# Save epi to a file
def save_rw2(reward2, filename):
    with open(filename, 'wb') as file:
        pickle.dump(reward2, file)

# Load epi from a file if it exists
def load_rw2(filename):
    try:
        with open(filename, 'rb') as file:
            reward2 = pickle.load(file)
            return reward2
    except FileNotFoundError:
        return None

def save_reset(reset, filename):
    with open(filename, 'wb') as file:
        pickle.dump(reset, file)
# Load epi from a file if it exists
def load_reset1(filename):
    try:
        with open(filename, 'rb') as file:
            reset1 = pickle.load(file)
            return reset1
    except FileNotFoundError:
        return None
# Load epi from a file if it exists
def load_reset2(filename):
    try:
        with open(filename, 'rb') as file:
            reset2 = pickle.load(file)
            return reset2
    except FileNotFoundError:
        return None

def save_eqrs(eqrs, filename):
    with open(filename, 'wb') as file:
        pickle.dump(eqrs, file)
def load_eqrs(filename):
    try:
        with open(filename, 'rb') as file:
            eqrs = pickle.load(file)
            return eqrs
    except FileNotFoundError:
        return None
        
#if os.path.exists(epi_filename):
#  epi = load_epi(epi_filename)
#else:
epi = 0.01
print(f"Loaded Epi is {epi}")

#if os.path.exists(epi_filename2):
#  epi2 = load_epi2(epi_filename2)
#else:
epi2 = 0.01
print(f"Loaded Epi2 is {epi2}")

if os.path.exists(rw_filename):
  reward = load_rw(rw_filename)
else:
  reward = 0
print(f"Loaded Reward is {reward}")

if os.path.exists(rw_filename2):
  reward2 = load_rw(rw_filename2)
else:
  reward2 = 0
print(f"Loaded Reward2 is {reward2}")

if os.path.exists(sc_filename):
  sell_count = load_sc(sc_filename)
else:
  sell_count = 0
print(f"Sell_count is {sell_count}")
if os.path.exists(bc_filename):
  buy_count = load_bc(bc_filename)
else:
  buy_count = 0
print(f"Buy_count is {buy_count}")

if os.path.exists(reset1_filename):
  reset1 = load_reset1(reset1_filename)
else:
  reset1 = 0
print(f"Reset1 is {reset1}")
if os.path.exists(reset2_filename):
  reset2 = load_reset2(reset2_filename)
else:
  reset2 = 0
print(f"Reset2 is {reset2}")

eqrs = 0

total_reset = reset1+reset2
if total_reset == 0:
  reset_p1 = 0
  reset_p2 = 0
else:
  reset_p1 = reset1/total_reset
  reset_p2 = reset2/total_reset
reset_bias = abs(reset_p1-reset_p2)
reset_threshold = 0.10
print("Reset Bias is ", reset_bias)
if reward >= reward2:
  if reset_p1 > reset_p2 and reset_bias > reset_threshold:
     agent_switch = 2
  else:
    agent_switch = 1
else:
  if reset_p2 > reset_p1 and reset_bias > reset_threshold:
    agent_switch = 1
  else:
    agent_switch = 2
    
test_model(epi,epi2, data, train, reward, reward2, agent_switch)
