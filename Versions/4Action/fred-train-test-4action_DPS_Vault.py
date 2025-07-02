import os
import pandas as pd
import numpy as np
import random
from collections import deque
from collections import defaultdict
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
import math
import csv
import json
import ast
import hashlib
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import scipy.special
import statistics

max_mem = 96*30
mem_check = 96*21

open("q_values_log.csv", "w").close()
open("q_values2_log.csv", "w").close()
open("q_values_select_log.csv", "w").close()

open("q_values_log_test.csv", "w").close()
open("q_values2_log_test.csv", "w").close()
open("q_values_select_log_test.csv", "w").close()

q_val_list = []
q_val_list2 = []
q_val_list_select = []

open("epi_log.csv", "w").close()

# Store hashes of trained batches
similarity_threshold = 0.9
trained_batches_file = "trained_batches.pkl"
if os.path.exists(trained_batches_file):
    with open(trained_batches_file, "rb") as f:
        trained_batches = pickle.load(f)
else:
    trained_batches = {}

def hash_batch(batch):
    """Create a hash based on state-action pairs for uniqueness."""
    batch_signature = str([(str(sample[0]), sample[2]) for sample in batch])  # state-action pairs
    return hashlib.sha256(batch_signature.encode()).hexdigest()

def hybrid_similarity(vec1, vec2, alpha=0.7):
    cos_sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    l2_dist = np.linalg.norm(vec1 - vec2)
    norm_l2 = 1 / (1 + l2_dist)  # normalize to [0, 1]
    return alpha * cos_sim + (1 - alpha) * norm_l2

def compute_batch_vector(batch, target_length=1152, num_actions=4):
    states = []
    actions = []
    rewards = []
    dones = []
    timestamps = []

    for sample in batch:
        state, original_action, action, reward, next_state, total_rw, done, timestamp = sample
        states.append(state.flatten())
        actions.append(action)
        rewards.append(total_rw)
        dones.append(done)
        timestamps.append(timestamp)

    states = np.array(states)
    rewards = np.array(rewards)
    dones = np.array(dones)

    state_mean = np.mean(states, axis=0)
    state_std = np.std(states, axis=0)

    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)
    reward_max = np.max(rewards)
    reward_min = np.min(rewards)
    reward_diff = rewards[-1] - rewards[0]
    reward_direction = np.sign(reward_diff)

    reward_stats = [reward_mean, reward_std, reward_max, reward_min, reward_diff, reward_direction]
    done_ratio = [np.mean(dones)]

    # Action distribution and entropy
    action_dist = np.bincount(actions, minlength=num_actions)
    action_dist_norm = action_dist / np.sum(action_dist)
    action_entropy = entropy(action_dist_norm, base=2)
    unique_action_count = len(np.unique(actions))

    # Normalize timestamps
    timestamps = np.array(timestamps).astype(float)
    t_min = np.min(timestamps)
    t_max = np.max(timestamps)
    if t_max - t_min > 0:
        timestamps = (timestamps - t_min) / (t_max - t_min)
    else:
        timestamps = timestamps * 0

    timestamp_stats = [
        np.mean(timestamps),
        np.std(timestamps),
        np.min(timestamps),
        np.max(timestamps)
    ]

    # Final vector composition
    vec = np.concatenate([
        state_mean,
        state_std,
        reward_stats,
        done_ratio,
        action_dist_norm,
        [action_entropy, unique_action_count],
        timestamp_stats
    ])

    if len(vec) < target_length:
        vec = np.pad(vec, (0, target_length - len(vec)), mode='constant')
    else:
        vec = vec[:target_length]

    return vec

def compute_and_save_market_stats():
    # List of all market feature column names (must match order in get_state())
    market_feature_columns = [
        'Close', 'MA_1', 'MA_2', 'MA_3', 'MA_4', 'MA_5', 'MA_6', 'MA_7', 'MA_8', 'MA_9',
        'MA_10', 'MA_11', 'RSI4', 'RSI8', 'RSI16', 'RSI32', 'RSI48', 'RSI96', 'RSI192',
        'RSI288', 'RSI384', 'RSI480', 'Short_MA', 'Long_MA', 'MACD', 'Signal_Line',
        'Rolling_Mean4', 'Upper_Band4', 'Lower_Band4', 'Band_Difference4',
        'Rolling_Mean8', 'Upper_Band8', 'Lower_Band8', 'Band_Difference8',
        'Rolling_Mean16', 'Upper_Band16', 'Lower_Band16', 'Band_Difference16',
        'Rolling_Mean32', 'Upper_Band32', 'Lower_Band32', 'Band_Difference32',
        'Rolling_Mean48', 'Upper_Band48', 'Lower_Band48', 'Band_Difference48',
        'Rolling_Mean96', 'Upper_Band96', 'Lower_Band96', 'Band_Difference96',
        'Rolling_Mean192', 'Upper_Band192', 'Lower_Band192', 'Band_Difference192',
        'Rolling_Mean288', 'Upper_Band288', 'Lower_Band288', 'Band_Difference288',
        'Rolling_Mean384', 'Upper_Band384', 'Lower_Band384', 'Band_Difference384',
        'Rolling_Mean480', 'Upper_Band480', 'Lower_Band480', 'Band_Difference480',
        'Lowest_Low', 'Highest_High', '%K', '%D', 'ROC4', 'ROC8', 'ROC16', 'ROC32',
        'ROC48', 'ROC96', 'ROC192', 'ROC288', 'ROC384', 'ROC480', 'TR4', 'ATR4', 'TR8',
        'ATR8', 'TR16', 'ATR16', 'TR32', 'ATR32', 'TR48', 'ATR48', 'TR96', 'ATR96',
        'TR192', 'ATR192', 'TR288', 'ATR288', 'TR384', 'ATR384', 'TR480', 'ATR480',
        'Pip_Range4', 'Smoothed_Pip_Range4', 'Pip_Range8', 'Smoothed_Pip_Range8',
        'Pip_Range16', 'Smoothed_Pip_Range16', 'Pip_Range32', 'Smoothed_Pip_Range32',
        'Pip_Range48', 'Smoothed_Pip_Range48', 'Pip_Range96', 'Smoothed_Pip_Range96',
        'Pip_Range192', 'Smoothed_Pip_Range192', 'Pip_Range288', 'Smoothed_Pip_Range288',
        'Pip_Range384', 'Smoothed_Pip_Range384', 'Pip_Range480', 'Smoothed_Pip_Range480',
        'Volume_Direction', 'OBV', '+DM', '-DM', '+DI', '-DI', 'DX', 'ADX'
    ]

    # Extract and clean market feature matrix from your dataset
    market_matrix = data[market_feature_columns].dropna().values

    # Compute stats
    market_mean = np.mean(market_matrix, axis=0)
    market_std = np.std(market_matrix, axis=0) + 1e-6  # avoid division by zero

    # Save to file
    np.save("market_mean.npy", market_mean)
    np.save("market_std.npy", market_std)

    print("Market feature mean and std saved to disk.")

def spread_indices(candidates, target, min_spacing):
    selected = []
    tries = 0
    max_tries = 50 * target  # prevent infinite loops
    while len(selected) < target and tries < max_tries:
        candidate = np.random.choice(candidates)
        if all(abs(candidate - idx) >= min_spacing for idx in selected):
            selected.append(candidate)
        tries += 1
    if len(selected) < target:
        # fallback to random with replacement
        additional = np.random.choice(candidates, size=target - len(selected), replace=True)
        selected.extend(additional)
    return selected

def stratified_random_batch(memory_list, batch_size, jitter_strength=0.3):
    from collections import defaultdict
    import numpy as np

    action_groups = defaultdict(list)

    # Group samples by action
    for idx, item in enumerate(memory_list):
        action = item[2]
        if action in [0, 1, 4, 5]:  # Focus only on key actions
            action_groups[action].append(idx)

    # Calculate base count per action
    base = batch_size // 4
    jitter_range = int(base * jitter_strength)

    # Create a jittered sample count for each action
    jittered_counts = {}
    total_requested = 0
    for action in [0, 1, 4, 5]:
        count = np.random.randint(base - jitter_range, base + jitter_range + 1)
        jittered_counts[action] = count
        total_requested += count

    # Scale if total goes over batch_size
    if total_requested > batch_size:
        scale = batch_size / total_requested
        for action in jittered_counts:
            jittered_counts[action] = max(1, int(jittered_counts[action] * scale))

    selected_indices = []

    for action in [0, 1, 4, 5]:
        target = jittered_counts[action]
        candidates = action_groups.get(action, [])
        if len(candidates) >= target:
            min_spacing = len(memory_list) // batch_size  # e.g., force distance of ~10 if batch is 128
            selected = spread_indices(candidates, target, min_spacing)
        else:
            selected = np.random.choice(candidates, size=target, replace=True) if candidates else []
        selected_indices.extend(selected)

    # Fill remainder if needed
    if len(selected_indices) < batch_size:
        remaining = batch_size - len(selected_indices)
        all_indices = list(range(len(memory_list)))
        filler = np.random.choice(all_indices, size=remaining, replace=False)
        selected_indices.extend(filler)

    selected_indices = sorted(selected_indices)
    batch = [memory_list[i] for i in selected_indices]

    return batch

def test_model(epi, epi2, data, train, reward, reward2, agent_switch):
    # Use an absolute file path
    data = pd.read_csv(r'data\data_2025.csv')

    # Remove unnecessary columns (open, high, low, vol, spread)
    data = data[['Date', 'Time', 'TickVol', 'High', 'Low', 'Close']]

    # Parse Date column and set it as the index
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data.set_index('Datetime', inplace=True)
    data.drop(columns=['Date'], inplace=True)  # Drop Date column after creating Datetime index

    moving_averages = [48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528]
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
    
    rsi_windows = [48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528]
    data['RSI4'] = calculate_rsi(data, window=rsi_windows[0])
    data['RSI8'] = calculate_rsi(data, window=rsi_windows[1])
    data['RSI16'] = calculate_rsi(data, window=rsi_windows[2])
    data['RSI32'] = calculate_rsi(data, window=rsi_windows[3])
    data['RSI48'] = calculate_rsi(data, window=rsi_windows[4])
    data['RSI96'] = calculate_rsi(data, window=rsi_windows[5])
    data['RSI192'] = calculate_rsi(data, window=rsi_windows[6])
    data['RSI288'] = calculate_rsi(data, window=rsi_windows[7])
    data['RSI384'] = calculate_rsi(data, window=rsi_windows[8])
    data['RSI480'] = calculate_rsi(data, window=rsi_windows[9])

    # Calculate MACD
    short_window = 48
    long_window = 192
    signal_window = 32
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    data['MACD'] = data['Short_MA'] - data['Long_MA']
    data['Signal_Line'] = data['MACD'].rolling(window=signal_window).mean()
    
    # Calculate Bollinger Bands
    window = 48
    data['Rolling_Mean4'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band4'] = data['Rolling_Mean4'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band4'] = data['Rolling_Mean4'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference4'] = data['Upper_Band4'] - data['Lower_Band4']
    
    # Calculate Bollinger Bands
    window = 96
    data['Rolling_Mean8'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band8'] = data['Rolling_Mean8'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band8'] = data['Rolling_Mean8'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference8'] = data['Upper_Band8'] - data['Lower_Band8']
    
    # Calculate Bollinger Bands
    window = 144
    data['Rolling_Mean16'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band16'] = data['Rolling_Mean16'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band16'] = data['Rolling_Mean16'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference16'] = data['Upper_Band16'] - data['Lower_Band16']
    
    # Calculate Bollinger Bands
    window = 192
    data['Rolling_Mean32'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band32'] = data['Rolling_Mean32'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band32'] = data['Rolling_Mean32'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference32'] = data['Upper_Band32'] - data['Lower_Band32']
    
    # Calculate Bollinger Bands
    window = 240
    data['Rolling_Mean48'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band48'] = data['Rolling_Mean48'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band48'] = data['Rolling_Mean48'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference48'] = data['Upper_Band48'] - data['Lower_Band48']
    
    # Calculate Bollinger Bands
    window = 288
    data['Rolling_Mean96'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band96'] = data['Rolling_Mean96'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band96'] = data['Rolling_Mean96'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference96'] = data['Upper_Band96'] - data['Lower_Band96']
    
    # Calculate Bollinger Bands
    window = 336
    data['Rolling_Mean192'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band192'] = data['Rolling_Mean192'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band192'] = data['Rolling_Mean192'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference192'] = data['Upper_Band192'] - data['Lower_Band192']
    
    # Calculate Bollinger Bands
    window = 384
    data['Rolling_Mean288'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band288'] = data['Rolling_Mean288'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band288'] = data['Rolling_Mean288'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference288'] = data['Upper_Band288'] - data['Lower_Band288']
    
    # Calculate Bollinger Bands
    window = 432
    data['Rolling_Mean384'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band384'] = data['Rolling_Mean384'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band384'] = data['Rolling_Mean384'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference384'] = data['Upper_Band384'] - data['Lower_Band384']
    
    # Calculate Bollinger Bands
    window = 480
    data['Rolling_Mean480'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band480'] = data['Rolling_Mean480'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band480'] = data['Rolling_Mean480'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference480'] = data['Upper_Band480'] - data['Lower_Band480']

    # Calculate Stochastic Oscillator
    k_window = 96
    d_window = 32
    data['Lowest_Low'] = data['Low'].rolling(window=k_window).min()
    data['Highest_High'] = data['High'].rolling(window=k_window).max()
    data['%K'] = (data['Close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low']) * 100
    data['%D'] = data['%K'].rolling(window=d_window).mean()

    # Calculate Price Rate of Change (ROC)
    roc_window = 48
    data['ROC4'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100
    roc_window = 96
    data['ROC8'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100
    roc_window = 144
    data['ROC16'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100
    roc_window = 192
    data['ROC32'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100
    roc_window = 240
    data['ROC48'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100
    roc_window = 288
    data['ROC96'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100
    roc_window = 336
    data['ROC192'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100
    roc_window = 384
    data['ROC288'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100
    roc_window = 432
    data['ROC384'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100
    roc_window = 480
    data['ROC480'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100

    # Calculate Average True Range (ATR)
    atr_window = 48
    data['TR4'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR4'] = data['TR4'].rolling(window=atr_window).mean()
    atr_window = 96
    data['TR8'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR8'] = data['TR8'].rolling(window=atr_window).mean()
    atr_window = 144
    data['TR16'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR16'] = data['TR16'].rolling(window=atr_window).mean()
    atr_window = 192
    data['TR32'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR32'] = data['TR32'].rolling(window=atr_window).mean()
    atr_window = 240
    data['TR48'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR48'] = data['TR48'].rolling(window=atr_window).mean()
    atr_window = 288
    data['TR96'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR96'] = data['TR96'].rolling(window=atr_window).mean()
    atr_window = 336
    data['TR192'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR192'] = data['TR192'].rolling(window=atr_window).mean()
    atr_window = 384
    data['TR288'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR288'] = data['TR288'].rolling(window=atr_window).mean()
    atr_window = 432
    data['TR384'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR384'] = data['TR384'].rolling(window=atr_window).mean()
    atr_window = 480
    data['TR480'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR480'] = data['TR480'].rolling(window=atr_window).mean()
    
    data['Pip_Range4'] = (data['Band_Difference4'] + data['ATR4']) / 2
    data['Smoothed_Pip_Range4'] = data['Pip_Range4'].rolling(window=48).mean()
    
    data['Pip_Range8'] = (data['Band_Difference8'] + data['ATR8']) / 2
    data['Smoothed_Pip_Range8'] = data['Pip_Range8'].rolling(window=96).mean()
    
    data['Pip_Range16'] = (data['Band_Difference16'] + data['ATR16']) / 2
    data['Smoothed_Pip_Range16'] = data['Pip_Range16'].rolling(window=144).mean()
    
    data['Pip_Range32'] = (data['Band_Difference32'] + data['ATR32']) / 2
    data['Smoothed_Pip_Range32'] = data['Pip_Range32'].rolling(window=192).mean()
    
    data['Pip_Range48'] = (data['Band_Difference48'] + data['ATR48']) / 2
    data['Smoothed_Pip_Range48'] = data['Pip_Range48'].rolling(window=240).mean()
    
    data['Pip_Range96'] = (data['Band_Difference96'] + data['ATR96']) / 2
    data['Smoothed_Pip_Range96'] = data['Pip_Range96'].rolling(window=288).mean()
    
    data['Pip_Range192'] = (data['Band_Difference192'] + data['ATR192']) / 2
    data['Smoothed_Pip_Range192'] = data['Pip_Range192'].rolling(window=336).mean()
    
    data['Pip_Range288'] = (data['Band_Difference288'] + data['ATR288']) / 2
    data['Smoothed_Pip_Range288'] = data['Pip_Range288'].rolling(window=384).mean()
    
    data['Pip_Range384'] = (data['Band_Difference384'] + data['ATR384']) / 2
    data['Smoothed_Pip_Range384'] = data['Pip_Range384'].rolling(window=432).mean()
    
    data['Pip_Range480'] = (data['Band_Difference480'] + data['ATR480']) / 2
    data['Smoothed_Pip_Range480'] = data['Pip_Range480'].rolling(window=480).mean()

    # Calculate On-Balance Volume (OBV)
    data['Volume_Direction'] = data['TickVol'].apply(lambda x: 1 if x >= 0 else -1)
    data['OBV'] = data['Volume_Direction'] * data['TickVol']
    data['OBV'] = data['OBV'].cumsum()
    data['OBV'] = data['OBV'].rolling(window=32).mean()

    # Calculate Average Directional Index (ADX)
    adx_window = 96
    data['High_Low'] = data['High'] - data['Low']
    data['High_Prev_Close'] = abs(data['High'] - data['Close'].shift(1))
    data['Low_Prev_Close'] = abs(data['Low'] - data['Close'].shift(1))
    data['+DM'] = data[['High_Low', 'High_Prev_Close']].apply(lambda x: x['High_Low'] if x['High_Low'] > x['High_Prev_Close'] else 0, axis=1)
    data['-DM'] = data[['High_Low', 'Low_Prev_Close']].apply(lambda x: x['Low_Prev_Close'] if x['Low_Prev_Close'] > x['High_Low'] else 0, axis=1)
    data['+DI'] = (data['+DM'].rolling(window=adx_window).mean() / data['ATR96'].rolling(window=adx_window).mean()) * 100
    data['-DI'] = (data['-DM'].rolling(window=adx_window).mean() / data['ATR96'].rolling(window=adx_window).mean()) * 100
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
    compute_and_save_market_stats()

    class TradingEnvironment:
        def __init__(self, data):
            self.data = data
            self.current_step = 0
            self.fee_rate = 0.10
            self.lifespan = 0
            self.prev_eqr = 0
            self.positions = []  # Dictionary to track open buy and sell options
            self.profit_history = [] 
            self.positive_count = 0
            self.negative_count = 0
            self.avg_check_eqr_list = []
            self.avg_check_eqr = 0
            self.npo_hp = 0
            self.eqr_switch = False
            self.hold_count = 0
            self.last_boost_step = -10 
            self.last_open_action = None
            self.range_mode = False
            self.portfolio_keys = [  # must match order in your portfolio_state array
                'total_balance', 'total_free_margin', 'total_equity', 'total_buy_options', 'total_sell_options',
                'sell_count', 'buy_count', 'start_hf', 'hold_fct', 'start_port', 'top_eqr', 'btm_eqr', 'check_bet',
                'check_eqr', 'action_factor', 'start_risk_ratio', 'start_bet', 'hf', 'end_port', 'reward_port',
                'rwp', 'holdings', 'pr', 'total_rw', 'risk_penalty', 'end_bet', 'bet', 'eqr', 'erw',
                'sc_per', 'bc_per', 'count_dist', 'trade_bias', 'pos_rw', 'reward', 'reward2', 'lifespan',
                'percent_positive', 'percent_negative', 'target_tier', 'avg_check_eqr', 'recent_losses', 'vault'
            ]
            self.categorical_portfolio_keys = {'target_tier', 'action_factor', 'trade_bias'}
            self.portfolio_mins = np.full(len(self.portfolio_keys), np.inf)
            self.portfolio_maxs = np.full(len(self.portfolio_keys), -np.inf)
            self.portfolio_normalization_config = None
            self.market_mean = np.load("market_mean.npy")
            self.market_std = np.load("market_std.npy")
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
                'top_eqr': 0.05,
                'btm_eqr': -0.05,
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
                'lifespan': 0,
                'percent_positive': 0,
                'percent_negative': 0,
                'target_tier': 0,
                'avg_check_eqr': 0,
                'recent_losses': 0,
                'vault': 0
                #'avg_check_eqr_list': [],
                #'trailing_stop': 0
            }


        def reset(self):
            self.current_step = 0
            self.positions = []  # Reset positions dictionary
            self.profit_history = []
            self.positive_count = 0
            self.negative_count = 0
            self.avg_check_eqr_list = []
            self.avg_check_eqr = 0
            #self.trailing_stop = 0
            self.npo_hp = 0
            self.eqr_switch = False
            self.hold_count = 0
            self.last_open_action = None
            self.range_mode = False
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
                'top_eqr': 0.05,
                'btm_eqr': -0.05,
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
                'lifespan': 0,
                'percent_positive': 0,
                'percent_negative': 0,
                'target_tier': 0,
                'avg_check_eqr': 0,
                'recent_losses': 0,
                'vault': 0
                #'avg_check_eqr_list': [],
                #'trailing_stop': 0
            }

        def save_portfolio_normalization_config(self, filename="portfolio_norm_config.json", buffer=0.05):
            config = {}

            for i, key in enumerate(self.portfolio_keys):
                low = float(self.portfolio_mins[i]) * (1 - buffer)
                high = float(self.portfolio_maxs[i]) * (1 + buffer)
                config[key] = [low, high]

            with open(filename, "w") as f:
                json.dump(config, f, indent=2)

            print(f"Saved portfolio normalization config to {filename}")
        
        def load_portfolio_normalization_config(self, filename="portfolio_norm_config.json"):
            with open(filename, "r") as f:
                self.portfolio_normalization_config = json.load(f)

            print(f"Loaded portfolio normalization config from {filename}")
            
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
                current_data['MA_7'],
                current_data['MA_8'],
                current_data['MA_9'],
                current_data['MA_10'],
                current_data['MA_11'],
                current_data['RSI4'],
                current_data['RSI8'],
                current_data['RSI16'],
                current_data['RSI32'],
                current_data['RSI48'],
                current_data['RSI96'],
                current_data['RSI192'],
                current_data['RSI288'],
                current_data['RSI384'],
                current_data['RSI480'],
                current_data['Short_MA'],
                current_data['Long_MA'],
                current_data['MACD'],
                current_data['Signal_Line'],
                current_data['Rolling_Mean4'],
                current_data['Upper_Band4'],
                current_data['Lower_Band4'],
                current_data['Band_Difference4'],
                current_data['Rolling_Mean8'],
                current_data['Upper_Band8'],
                current_data['Lower_Band8'],
                current_data['Band_Difference8'],
                current_data['Rolling_Mean16'],
                current_data['Upper_Band16'],
                current_data['Lower_Band16'],
                current_data['Band_Difference16'],
                current_data['Rolling_Mean32'],
                current_data['Upper_Band32'],
                current_data['Lower_Band32'],
                current_data['Band_Difference32'],
                current_data['Rolling_Mean48'],
                current_data['Upper_Band48'],
                current_data['Lower_Band48'],
                current_data['Band_Difference48'],
                current_data['Rolling_Mean96'],
                current_data['Upper_Band96'],
                current_data['Lower_Band96'],
                current_data['Band_Difference96'],
                current_data['Rolling_Mean192'],
                current_data['Upper_Band192'],
                current_data['Lower_Band192'],
                current_data['Band_Difference192'],
                current_data['Rolling_Mean288'],
                current_data['Upper_Band288'],
                current_data['Lower_Band288'],
                current_data['Band_Difference288'],
                current_data['Rolling_Mean384'],
                current_data['Upper_Band384'],
                current_data['Lower_Band384'],
                current_data['Band_Difference384'],
                current_data['Rolling_Mean480'],
                current_data['Upper_Band480'],
                current_data['Lower_Band480'],
                current_data['Band_Difference480'],
                current_data['Lowest_Low'],
                current_data['Highest_High'],
                current_data['%K'],
                current_data['%D'],
                current_data['ROC4'],
                current_data['ROC8'],
                current_data['ROC16'],
                current_data['ROC32'],
                current_data['ROC48'],
                current_data['ROC96'],
                current_data['ROC192'],
                current_data['ROC288'],
                current_data['ROC384'],
                current_data['ROC480'],
                current_data['TR4'],
                current_data['ATR4'],
                current_data['TR8'],
                current_data['ATR8'],
                current_data['TR16'],
                current_data['ATR16'],
                current_data['TR32'],
                current_data['ATR32'],
                current_data['TR48'],
                current_data['ATR48'],
                current_data['TR96'],
                current_data['ATR96'],
                current_data['TR192'],
                current_data['ATR192'],
                current_data['TR288'],
                current_data['ATR288'],
                current_data['TR384'],
                current_data['ATR384'],
                current_data['TR480'],
                current_data['ATR480'],
                current_data['Pip_Range4'],
                current_data['Smoothed_Pip_Range4'],
                current_data['Pip_Range8'],
                current_data['Smoothed_Pip_Range8'],
                current_data['Pip_Range16'],
                current_data['Smoothed_Pip_Range16'],
                current_data['Pip_Range32'],
                current_data['Smoothed_Pip_Range32'],
                current_data['Pip_Range48'],
                current_data['Smoothed_Pip_Range48'],
                current_data['Pip_Range96'],
                current_data['Smoothed_Pip_Range96'],
                current_data['Pip_Range192'],
                current_data['Smoothed_Pip_Range192'],
                current_data['Pip_Range288'],
                current_data['Smoothed_Pip_Range288'],
                current_data['Pip_Range384'],
                current_data['Smoothed_Pip_Range384'],
                current_data['Pip_Range480'],
                current_data['Smoothed_Pip_Range480'],
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
            normalized_market = (current_features - self.market_mean) / (self.market_std + 1e-6)
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
                self.portfolio['lifespan'], 
                self.portfolio['percent_positive'],
                self.portfolio['percent_negative'],
                self.portfolio['target_tier'],
                self.portfolio['avg_check_eqr'],
                self.portfolio['recent_losses'],
                self.portfolio['vault']
                #self.portfolio['avg_check_eqr_list'],
                #self.portfolio['trailing_stop']
            ], dtype=np.float32)
            self.portfolio_mins = np.minimum(self.portfolio_mins, portfolio_state)
            self.portfolio_maxs = np.maximum(self.portfolio_maxs, portfolio_state)
            normalized_portfolio = []
            for i, (key, value) in enumerate(zip(self.portfolio_keys, portfolio_state)):
                if key in self.categorical_portfolio_keys:
                    norm_value = float(value)  # Pass through unnormalized (or one-hot outside loop)
                else:
                    if self.portfolio_normalization_config:
                        low, high = self.portfolio_normalization_config[key]
                    else:
                        low, high = self.portfolio_mins[i], self.portfolio_maxs[i]
                    norm_value = (value - low) / (high - low + 1e-6)
                    norm_value = np.clip(norm_value, 0, 1)
                normalized_portfolio.append(norm_value)

            normalized_portfolio = np.array(normalized_portfolio, dtype=np.float32)
            
            # Concatenate current features and portfolio information
            state = np.concatenate((normalized_market, normalized_portfolio))

            return state
    
    
        def step(self, action, reward, reward2, agent_switch, action_prob):
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
                if holding_period !=0:
                    holding_period_factor = (holding_period / max_holding_period)# Adjust this as needed
                else:
                    holding_period_factor = 0
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
            
            
            if len(self.positions) == 0: 
                self.profit_history = []
                self.positive_count = 0
                self.negative_count = 0
            else:
                # Store the profit in the history
                self.profit_history.append(total_open_positions_profit)
                # Update positive/negative counts
                if total_open_positions_profit > 0:
                    self.positive_count += 1
                else:
                    self.negative_count += 1
            print(f"Count of steps with profit above 0: {self.positive_count:.2f}")
            print(f"Count of steps with profit below or equal to 0: {self.negative_count:.2f}")
            # Calculate percentages
            if (self.positive_count + self.negative_count) != 0:
                 percent_positive = (self.positive_count / (self.positive_count + self.negative_count))
                 percent_negative = (self.negative_count / (self.positive_count + self.negative_count))
            else:
                percent_positive = 0
                percent_negative = 0
            self.portfolio['percent_positive'] = percent_positive
            self.portfolio['percent_negative'] = percent_negative
            # Output the results
            print(f"Percentage of steps with profit above 0: {percent_positive:.2f}%")
            print(f"Percentage of steps with profit below or equal to 0: {percent_negative:.2f}%")
            
            fee = 0
            prev_total_balance = self.portfolio['total_balance']
            if action == 0 or action == 1:
                base_pz = self.portfolio['total_free_margin'] / 1000.0
                if self.portfolio['recent_losses'] >= 1: 
                    action_prob *= 0.5
                scaled_confidence = max(0.01, min(action_prob, 1.0))  # clip to [0.01, 1.0]
                adjusted_pz = base_pz * scaled_confidence
                action_pz = min(max(1, adjusted_pz), 10)
            else:
                action_pz = 0
            pz = action_pz #self.portfolio['total_free_margin']/1000
            if pz > 10:
              pz = 10
            elif pz < 1:
              pz = 1
            print("The posiiton size is: ", pz)
            action_factor = 1
            #ATR = self.data['ATR8'][self.current_step]
            if len(self.positions) == 0:
              bet_scale = pz * (self.portfolio['total_free_margin']*0.01)
            else:
              bet_scale = (self.portfolio['total_balance']-self.portfolio['total_free_margin'])
            scaled_value4 = ((self.data['Smoothed_Pip_Range4'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value4 < 0.01:
              scaled_value4 = 0.01
            print("The scaled target 4 is ", scaled_value4)
            scaled_value8 = ((self.data['Smoothed_Pip_Range8'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value8 < 0.01:
              scaled_value8 = 0.01
            print("The scaled target 8 is ", scaled_value8)
            scaled_value16 = ((self.data['Smoothed_Pip_Range16'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value16 < 0.01:
              scaled_value16 = 0.01
            print("The scaled target 16 is ", scaled_value16)
            scaled_value32 = ((self.data['Smoothed_Pip_Range32'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value32 < 0.01:
              scaled_value32 = 0.01
            print("The scaled target 32 is ", scaled_value32)
            scaled_value48 = ((self.data['Smoothed_Pip_Range48'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value48 < 0.01:
              scaled_value48 = 0.01
            print("The scaled target 48 is ", scaled_value48)
            scaled_value96 = ((self.data['Smoothed_Pip_Range96'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value96 < 0.01:
              scaled_value96 = 0.01
            print("The scaled target 96 is ", scaled_value96)
            scaled_value192 = ((self.data['Smoothed_Pip_Range192'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value192 < 0.01:
              scaled_value192 = 0.01
            print("The scaled target 192 is ", scaled_value192)
            scaled_value288 = ((self.data['Smoothed_Pip_Range288'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value288 < 0.01:
              scaled_value288 = 0.01
            print("The scaled target 288 is ", scaled_value288)
            scaled_value384 = ((self.data['Smoothed_Pip_Range384'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value384 < 0.01:
              scaled_value384 = 0.01
            print("The scaled target 384 is ", scaled_value384)
            scaled_value480 = ((self.data['Smoothed_Pip_Range480'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value480 < 0.01:
              scaled_value480 = 0.01
            print("The scaled target 480 is ", scaled_value480)
            
            base_rate = 0.1  # Minimum transition rate (10%)
            #volatility_factor = 2.0  # Sensitivity to volatility
            volatility_factor4 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))
            volatility_factor8 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))
            volatility_factor16 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))
            volatility_factor32 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))
            volatility_factor48 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))
            volatility_factor96 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))
            volatility_factor192 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))
            volatility_factor288 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))
            volatility_factor384 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))
            volatility_factor480 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))

            # Calculate dynamic transition rate for each scaled value
            transition_rate4 = min(max(base_rate + (volatility_factor4 * scaled_value4), 0.1), 0.5)
            transition_rate8 = min(max(base_rate + (volatility_factor8 * scaled_value8), 0.1), 0.5)
            transition_rate16 = min(max(base_rate + (volatility_factor16 * scaled_value16), 0.1), 0.5)
            transition_rate32 = min(max(base_rate + (volatility_factor32 * scaled_value32), 0.1), 0.5)
            transition_rate48 = min(max(base_rate + (volatility_factor48 * scaled_value48), 0.1), 0.5)
            transition_rate96 = min(max(base_rate + (volatility_factor96 * scaled_value96), 0.1), 0.5)
            transition_rate192 = min(max(base_rate + (volatility_factor192 * scaled_value192), 0.1), 0.5)
            transition_rate288 = min(max(base_rate + (volatility_factor288 * scaled_value288), 0.1), 0.5)
            transition_rate384 = min(max(base_rate + (volatility_factor384 * scaled_value384), 0.1), 0.5)
            transition_rate480 = min(max(base_rate + (volatility_factor480 * scaled_value480), 0.1), 0.5)

            #top_eqr = (2)*scaled_value
            #btm_eqr = (-1)*scaled_value
            tfmc = self.portfolio['total_free_margin']
            check_bet = (self.portfolio['total_balance']-tfmc)
            if start_port != 0 and check_bet != 0:
              check_eqr = start_port/check_bet
            else:
              check_eqr = 0
              
            if len(self.positions) == 0: 
                self.avg_check_eqr_list = []
                self.avg_check_eqr = 0
            else:
                self.avg_check_eqr_list.append(check_eqr)
                if sum(self.avg_check_eqr_list) == 0 or len(self.avg_check_eqr_list) == 0:
                    self.avg_check_eqr = 0
                else:
                    # Number of values in the list
                    n = len(self.avg_check_eqr_list)
        
                    # Base decay rate and dynamic scaling
                    base_rate = 0.05  # Adjust this as needed
                    decay_rate = base_rate * math.log(n + 1)  # Logarithmic scaling
        
                    # Exponential decay for the newest weight
                    newest_weight = math.exp(-decay_rate * n)

                    # Calculate remaining weight
                    remaining_weight = 1 - newest_weight

                    # Descending proportions for older values
                    descending_proportions = list(range(n - 1, 0, -1))  # Example: [4, 3, 2, 1] for n=5
                    total_proportions = sum(descending_proportions)

                    # Calculate weights for older values
                    additional_weights = [
                        remaining_weight * (p / total_proportions) for p in descending_proportions
                    ]

                    # Combine weights with the newest value
                    weights = additional_weights + [newest_weight]

                    # Reverse weights so the newest (last value) has the highest weight
                    weights.reverse()

                    # Calculate weighted mean
                    weighted_mean = sum(value * weight for value, weight in zip(self.avg_check_eqr_list, weights))
        
                    # Update the average
                    self.avg_check_eqr = weighted_mean
            self.portfolio['avg_check_eqr'] = self.avg_check_eqr
            #self.portfolio['avg_check_eqr_list'] = self.avg_check_eqr_list
            print(f"The average eqr is: {self.avg_check_eqr}")
            
            if len(self.avg_check_eqr_list) >= 1:
                median_eqr = statistics.median([abs(x) for x in self.avg_check_eqr_list])
            else:
                median_eqr = 0
            print(f"The median eqr is: {median_eqr}")
            
            if len(self.positions) == 0:
                top_eqr = scaled_value4
                btm_eqr = (scaled_value4*-1)/2
                if scaled_value4 < 0.20:
                    self.range_mode = True
            else:
                top_eqr = self.portfolio['top_eqr']
                btm_eqr = self.portfolio['btm_eqr']
            self.portfolio['top_eqr'] = top_eqr
            self.portfolio['btm_eqr'] = btm_eqr
            ttb1 = btm_eqr
            ttb2 = btm_eqr
            ttb3 = btm_eqr
            ttb4 = btm_eqr
            ttb5 = btm_eqr
            ttb6 = btm_eqr
            ttb7 = btm_eqr
            ttb8 = btm_eqr
            ttb9 = btm_eqr
            ttb10 = btm_eqr
            if len(self.positions)==0:
                 if self.eqr_switch == True:
                     self.eqr_switch = False
                 if self.range_mode == True and scaled_value4 >= 0.20:
                     self.range_mode = False
            if self.portfolio['lifespan'] % 4 == 0:
                if median_eqr < 0.10 and scaled_value4 < 0.20 and self.portfolio['lifespan'] % 48 == 0 and self.portfolio['lifespan'] != 0:
                    top_eqr = self.avg_check_eqr
                    btm_eqr = btm_eqr
                    self.range_mode = True
                    print("Ranging Mode...")
                else:
                    print("Trending Mode...")
                    if 4 <= self.portfolio['lifespan'] < 32 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate4) * top_eqr + transition_rate4 * scaled_value4
                         btm_eqr = top_eqr / -2
                         ttb1 = btm_eqr
                         self.portfolio['target_tier'] = 1
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
                    elif 32 <= self.portfolio['lifespan'] < 64 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate8) * top_eqr + transition_rate8 * scaled_value8
                         if btm_eqr >= 0:
                             if self.eqr_switch == False:
                                 btm_eqr = min((1 - transition_rate8) * btm_eqr + transition_rate8 * scaled_value8, (top_eqr/2))
                                 print("Positive transition btm_eqr is: ", btm_eqr)
                             else:
                                 print("No Positive transition eqr switch True  btm_eqr is: ", btm_eqr)
                         else:
                             check_btm = (1 - transition_rate8) * btm_eqr + transition_rate8 * scaled_value8
                             if check_eqr < check_btm:
                                 if check_eqr > top_eqr/2:
                                     btm_eqr = btm_eqr/2
                                 else:
                                     btm_eqr = btm_eqr
                             else: 
                                btm_eqr = check_btm
                             print("Negtive transition btm_eqr is: ", btm_eqr)
                         if -0.01 < btm_eqr < 0:
                             btm_eqr = 0
                         ttb2 = btm_eqr
                         self.portfolio['target_tier'] = 2
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
                    elif 64 <= self.portfolio['lifespan'] < 96 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate16) * top_eqr + transition_rate16 * scaled_value16
                         if btm_eqr >= 0:
                             if self.eqr_switch == False:
                                 btm_eqr = min((1 - transition_rate16) * btm_eqr + transition_rate16 * scaled_value16, (top_eqr/2))
                                 print("Positive transition btm_eqr is: ", btm_eqr)
                             else:
                                 print("No Positive transition eqr switch True  btm_eqr is: ", btm_eqr)
                         else:
                             check_btm = (1 - transition_rate16) * btm_eqr + transition_rate16 * scaled_value16
                             if check_eqr < check_btm:
                                 if check_eqr > top_eqr/2:
                                     btm_eqr = btm_eqr/2
                                 else:
                                     btm_eqr = btm_eqr
                             else: 
                                btm_eqr = check_btm
                             print("Negtive transition btm_eqr is: ", btm_eqr)
                         if -0.01 < btm_eqr < 0:
                             btm_eqr = 0
                         ttb3 = btm_eqr
                         self.portfolio['target_tier'] = 3
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
                    elif 96 <= self.portfolio['lifespan'] < 128 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate32) * top_eqr + transition_rate32 * scaled_value32
                         if btm_eqr >= 0:
                             if self.eqr_switch == False:
                                 btm_eqr = min((1 - transition_rate32) * btm_eqr + transition_rate32 * scaled_value32, (top_eqr/2))
                                 print("Positive transition btm_eqr is: ", btm_eqr)
                             else:
                                 print("No Positive transition eqr switch True  btm_eqr is: ", btm_eqr)
                         else:
                             check_btm = (1 - transition_rate32) * btm_eqr + transition_rate32 * scaled_value32
                             if check_eqr < check_btm:
                                 if check_eqr > top_eqr/2:
                                     btm_eqr = btm_eqr/2
                                 else:
                                     btm_eqr = btm_eqr
                             else: 
                                btm_eqr = check_btm
                             print("Negtive transition btm_eqr is: ", btm_eqr)
                         if -0.01 < btm_eqr < 0:
                             btm_eqr = 0
                         ttb4 = btm_eqr
                         self.portfolio['target_tier'] = 4
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
                    elif 128 <= self.portfolio['lifespan'] < 160 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate48) * top_eqr + transition_rate48 * scaled_value48
                         if btm_eqr >= 0:
                             if self.eqr_switch == False:
                                 btm_eqr = min((1 - transition_rate48) * btm_eqr + transition_rate48 * scaled_value48, (top_eqr/2))
                                 print("Positive transition btm_eqr is: ", btm_eqr)
                             else:
                                 print("No Positive transition eqr switch True  btm_eqr is: ", btm_eqr)
                         else:
                             check_btm = (1 - transition_rate48) * btm_eqr + transition_rate48 * scaled_value48
                             if check_eqr < check_btm:
                                 if check_eqr > top_eqr/2:
                                     btm_eqr = btm_eqr/2
                                 else:
                                     btm_eqr = btm_eqr
                             else: 
                                btm_eqr = check_btm
                             print("Negtive transition btm_eqr is: ", btm_eqr)
                         if -0.01 < btm_eqr < 0:
                             btm_eqr = 0
                         ttb5 = btm_eqr
                         self.portfolio['target_tier'] = 5
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
                    elif 160 <= self.portfolio['lifespan'] < 192 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate96) * top_eqr + transition_rate96 * scaled_value96
                         if btm_eqr >= 0:
                             if self.eqr_switch == False:
                                 btm_eqr = min((1 - transition_rate96) * btm_eqr + transition_rate96 * scaled_value96, (top_eqr/2))
                                 print("Positive transition btm_eqr is: ", btm_eqr)
                             else:
                                 print("No Positive transition eqr switch True  btm_eqr is: ", btm_eqr)
                         else:
                             check_btm = (1 - transition_rate96) * btm_eqr + transition_rate96 * scaled_value96
                             if check_eqr < check_btm:
                                 if check_eqr > top_eqr/2:
                                     btm_eqr = btm_eqr/2
                                 else:
                                     btm_eqr = btm_eqr
                             else: 
                                btm_eqr = check_btm
                             print("Negtive transition btm_eqr is: ", btm_eqr)
                         if -0.01 < btm_eqr < 0:
                             btm_eqr = 0
                         ttb6 = btm_eqr
                         self.portfolio['target_tier'] = 6
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
                    elif  192 <= self.portfolio['lifespan'] < 224 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate192) * top_eqr + transition_rate192 * scaled_value192
                         if btm_eqr >= 0:
                             if self.eqr_switch == False:
                                 btm_eqr = min((1 - transition_rate192) * btm_eqr + transition_rate192 * scaled_value192, (top_eqr/2))
                                 print("Positive transition btm_eqr is: ", btm_eqr)
                             else:
                                 print("No Positive transition eqr switch True  btm_eqr is: ", btm_eqr)
                         else:
                             check_btm = (1 - transition_rate192) * btm_eqr + transition_rate192 * scaled_value192
                             if check_eqr < check_btm:
                                 if check_eqr > top_eqr/2:
                                     btm_eqr = btm_eqr/2
                                 else:
                                     btm_eqr = btm_eqr
                             else: 
                                btm_eqr = check_btm
                             print("Negtive transition btm_eqr is: ", btm_eqr)
                         if -0.01 < btm_eqr < 0:
                             btm_eqr = 0
                         ttb7 = btm_eqr
                         self.portfolio['target_tier'] = 7
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
                    elif  224 <= self.portfolio['lifespan'] < 256 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate288) * top_eqr + transition_rate288 * scaled_value288
                         if btm_eqr >= 0:
                             if self.eqr_switch == False:
                                 btm_eqr = min((1 - transition_rate288) * btm_eqr + transition_rate288 * scaled_value288, (top_eqr/2))
                                 print("Positive transition btm_eqr is: ", btm_eqr)
                             else:
                                 print("No Positive transition eqr switch True  btm_eqr is: ", btm_eqr)
                         else:
                             check_btm = (1 - transition_rate288) * btm_eqr + transition_rate288 * scaled_value288
                             if check_eqr < check_btm:
                                 if check_eqr > top_eqr/2:
                                     btm_eqr = btm_eqr/2
                                 else:
                                     btm_eqr = btm_eqr
                             else: 
                                btm_eqr = check_btm
                             print("Negtive transition btm_eqr is: ", btm_eqr)
                         if -0.01 < btm_eqr < 0:
                             btm_eqr = 0
                         ttb8 = btm_eqr
                         self.portfolio['target_tier'] = 8
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
                    elif  256 <= self.portfolio['lifespan'] < 288 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate384) * top_eqr + transition_rate384 * scaled_value384
                         if btm_eqr >= 0:
                             if self.eqr_switch == False:
                                 btm_eqr = min((1 - transition_rate384) * btm_eqr + transition_rate384 * scaled_value384, (top_eqr/2))
                                 print("Positive transition btm_eqr is: ", btm_eqr)
                             else:
                                 print("No Positive transition eqr switch True  btm_eqr is: ", btm_eqr)
                         else:
                             check_btm = (1 - transition_rate384) * btm_eqr + transition_rate384 * scaled_value384
                             if check_eqr < check_btm:
                                 if check_eqr > top_eqr/2:
                                     btm_eqr = btm_eqr/2
                                 else:
                                     btm_eqr = btm_eqr
                             else: 
                                btm_eqr = check_btm
                             print("Negtive transition btm_eqr is: ", btm_eqr)
                         if -0.01 < btm_eqr < 0:
                             btm_eqr = 0
                         ttb9 = btm_eqr
                         self.portfolio['target_tier'] = 9
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
                    elif  288 <= self.portfolio['lifespan'] < 320 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate480) * top_eqr + transition_rate480 * scaled_value480
                         if btm_eqr >= 0:
                             if self.eqr_switch == False:
                                 btm_eqr = min((1 - transition_rate480) * btm_eqr + transition_rate480 * scaled_value480, (top_eqr/2))
                                 print("Positive transition btm_eqr is: ", btm_eqr)
                             else:
                                 print("No Positive transition eqr switch True  btm_eqr is: ", btm_eqr)
                         else:
                             check_btm = (1 - transition_rate480) * btm_eqr + transition_rate480 * scaled_value480
                             if check_eqr < check_btm:
                                 if check_eqr > top_eqr/2:
                                     btm_eqr = btm_eqr/2
                                 else:
                                     btm_eqr = btm_eqr
                             else: 
                                btm_eqr = check_btm
                             print("Negtive transition btm_eqr is: ", btm_eqr)
                         if -0.01 < btm_eqr < 0:
                             btm_eqr = 0
                         ttb10 = btm_eqr
                         self.portfolio['target_tier'] = 10
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
            if check_eqr >= 1:
                 btm_eqr = max(self.avg_check_eqr*.9,btm_eqr)
                 self.eqr_switch = True
            self.portfolio['top_eqr'] = top_eqr
            self.portfolio['btm_eqr'] = btm_eqr
            print(f"Position Target Tier is: {self.portfolio['target_tier']}")
            print("Top Equity Target is: ", top_eqr)
            print("Bottom Equity Target is: ", btm_eqr)
            print("Range Mode is: ", self.range_mode)
            
            rsi = self.data['RSI8'][self.current_step]
            obv = self.data['OBV'][self.current_step]
            short_ma = self.data['Short_MA'][self.current_step]
            long_ma = self.data['Long_MA'][self.current_step]
            
            if action == 0 or action == 1:
                if len(self.positions) >= 1:
                    action = 4
                    
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
            
            if action == 5:
              tfmc = self.portfolio['total_free_margin']
              check_bet = (self.portfolio['total_balance']-tfmc)
              if start_port != 0 and check_bet != 0:
                check_eqr = start_port/check_bet
              else:
                check_eqr = 0
              if btm_eqr < check_eqr < top_eqr:
                action = 4
                action_factor = -1
            
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
              if eqrs == 1:
                if check_eqr < self.prev_eqr:
                  action = 5
                  eqrs = 0
                  save_eqrs(eqrs, eqrs_status_filename)
                  
            if self.range_mode == True:
                if check_eqr >= top_eqr:
                    action = 5
                    print("Range Mode force closing at top eqr...")
            if hold_fct >= 1:
                action = 5
                action_factor = -1
            if action == 0:
              buy_count += 1
            if action == 1:
              sell_count += 1

            save_sell_count(sell_count, sc_filename)
            save_buy_count(buy_count, bc_filename)
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
            
            elif action == 4:  # Hold position (do nothing)
                for position in self.positions:
                    if position['type'] == 'buy':
                         position['current_profit'] = ((self.data['Close'][self.current_step] - position['open_price'])*10000) * (position['position_size'])
                    elif position['type'] == 'sell':
                         position['current_profit'] = ((position['open_price'] - self.data['Close'][self.current_step])*10000) * (position['position_size'])

            elif action == 5:  # Close all open buy and sell options
                positions_to_remove = []
                if 0 <= check_eqr < .10:
                    retain_amt = .90
                    vault_dis = .10
                elif 0.10 <= check_eqr < .20:
                    retain_amt = .80
                    vault_dis = .20
                elif 0.20 <= check_eqr < .30:
                    retain_amt = .70
                    vault_dis = .30
                elif 0.30 <= check_eqr < .40:
                    retain_amt = .60
                    vault_dis = .40
                elif 0.40 <= check_eqr < .50:
                    retain_amt = .50
                    vault_dis = .50
                elif 0.50 <= check_eqr < .60:
                    retain_amt = .40
                    vault_dis = .60
                elif 0.60 <= check_eqr < .70:
                    retain_amt = .30
                    vault_dis = .70
                elif 0.70 <= check_eqr < .80:
                    retain_amt = .20
                    vault_dis = .80
                elif check_eqr >= .80:
                    retain_amt = .10
                    vault_dis = .90
                for position in self.positions:
                    if position['type'] == 'buy':
                        buy_profit = position['current_profit']
                        buy_profit_amt = position['current_profit']
                        if buy_profit > 0:
                            buy_profit_amt = buy_profit * retain_amt
                            vault_amt = buy_profit * vault_dis
                            self.portfolio['vault'] += vault_amt
                        self.portfolio['total_balance'] += buy_profit_amt
                        self.portfolio['total_free_margin'] += buy_profit_amt
                        self.portfolio['total_free_margin'] += position['total_margin']
                        #self.positions.remove(position)
                        self.portfolio['total_buy_options'] -= 1
                    elif position['type'] == 'sell':
                        sell_profit = position['current_profit']
                        sell_profit_amt = position['current_profit']
                        if sell_profit > 0:
                            sell_profit_amt = sell_profit * retain_amt
                            vault_amt = sell_profit * vault_dis
                            self.portfolio['vault'] += vault_amt
                        self.portfolio['total_balance'] += sell_profit_amt
                        self.portfolio['total_free_margin'] += sell_profit_amt
                        self.portfolio['total_free_margin'] += position['total_margin']
                        #self.positions.remove(position)
                        self.portfolio['total_sell_options'] -= 1
                
                    positions_to_remove.append(position)
                
                for position in positions_to_remove:
                    self.positions.remove(position) 
            
            if self.positions:
                print("Position size:", self.positions[0]['position_size'])
            else:
                print("No open position yet.")
       
            self.portfolio['total_equity'] = self.portfolio['total_balance']
            for position in self.positions:
                pt = position['current_profit']
                self.portfolio['total_equity'] += pt
            print("-------------------------------------------------------------")
            #positioning life span 
            if len(self.positions) == 0:
              if action == 5:
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
                if holding_period !=0:
                    holding_period_factor = (holding_period / max_holding_period)# Adjust this as needed
                else:
                    holding_period_factor = 0
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
            if start_port != 0 and end_port != 0:
                reward_port = (end_port + start_port)/2
            else:
                reward_port = (end_port + start_port)
            self.portfolio['end_port'] = end_port
            if self.portfolio['reward_port'] != 0 and start_bet != 0:
                self.portfolio['reward_port'] = reward_port/start_bet
            else:
                self.portfolio['reward_port'] = reward_port
            rwp = 0
            if reward_port > 0:
              if agent_switch == 1:
                if len(self.positions)==0:
                  reward += 0
                  rwp += 0
                else:
                  if reward_port != 0 and start_bet != 0:
                      reward += reward_port/start_bet
                      rwp += reward_port/start_bet
                  else: 
                      reward += 0
                      rwp += 0
              else:
                if len(self.positions)==0:
                  reward2 += 0
                  rwp += 0
                else:
                  if reward_port != 0 and start_bet != 0:
                      reward2 += reward_port/start_bet
                      rwp += reward_port/start_bet
                  else: 
                      reward2 += 0
                      rwp += 0
            elif reward_port < 0:
              if agent_switch == 1:
                if len(self.positions)==0:
                   reward += 0
                   rwp += 0
                else:
                  if reward_port != 0 and start_bet != 0:
                      reward += reward_port/start_bet
                      rwp += reward_port/start_bet
                  else: 
                      reward += 0
                      rwp += 0
              else:
                if len(self.positions)==0:
                   reward2 += 0
                   rwp += 0
                else:
                  if reward_port != 0 and start_bet != 0:
                      reward2 += reward_port/start_bet
                      rwp += reward_port/start_bet
                  else: 
                      reward2 += 0
                      rwp += 0
            else:
              reward += 0
              reward2 += 0
            print(f"Total Equity Reward/Penalty: {rwp}")
            self.portfolio['rwp'] = rwp
            # Portfolio Rewards ----
            holdings = 0
            hdr_diff = (self.portfolio['total_balance']-10000)
            if hdr_diff != 0:
                hdr = hdr_diff / 10000
            else:
                hdr = 0
            if self.portfolio['total_balance'] < 10000:
                if agent_switch == 1:
                  if len(self.positions)==0:
                     holdings += 0
                     reward += 0
                  else:
                    holdings += hdr
                    reward += hdr
                else:
                  if len(self.positions)==0:
                     holdings += 0
                     reward2 += 0
                  else:
                    holdings += hdr
                    reward2 += hdr
            elif self.portfolio['total_balance'] >= 10000:
                if agent_switch == 1:
                  if len(self.positions)==0:
                     holdings += 0
                     reward += 0
                  else:
                    holdings += hdr
                    reward += hdr
                else:
                  if len(self.positions)==0:
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
            port_return_rate = portfolio_reward/(start_bet+ 1e-6)
            # Update the reward based on the change in total balance
            pr = 0
            if portfolio_reward != 0:
                port_rd = (port_return_rate / (self.portfolio['lifespan'] + 1e-6)) + (portfolio_reward * 0.0001)
            else:
                port_rd = 0
            if portfolio_reward > 0:
              if agent_switch == 1:
                if len(self.positions)==0:
                   pr += 0
                   reward += 0
                else:
                   pr += port_rd 
                   reward += port_rd 
              else:
                if len(self.positions)==0:
                   pr += 0
                   reward2 += 0
                else:
                  pr += port_rd 
                  reward2 += port_rd 
            elif portfolio_reward < 0:
              if agent_switch == 1:
                if len(self.positions)==0:
                   pr += 0
                   reward += 0
                else:
                  pr += port_rd
                  reward += port_rd
              else:
                if len(self.positions)==0:
                   pr += 0
                   reward2 += 0
                else:
                  pr += port_rd
                  reward2 += port_rd
            else:
              reward += 0
              reward2 += 0
            print("Closing Portfolio Balance Reward/Penalty:", pr)
            self.portfolio['pr'] = pr
            afrw = 0
            if action_factor == 1:
              if agent_switch == 1:
                if len(self.positions)==0:
                   reward += 0
                   afrw +=0
                else:
                  reward += 0.01
                  afrw += 0.01
              else:
                if len(self.positions)==0:
                   reward2 += 0
                   afrw += 0
                else:
                  reward2 += 0.01
                  afrw += 0.01
            else:
              if agent_switch == 1:
                if len(self.positions)==0:
                   reward += 0
                   afrw += 0
                else:
                  reward += -0.01
                  afrw += -0.01
              else:
                if len(self.positions)==0:
                   reward += 0
                   afrw += 0
                else:
                  afrw += -0.01
                  reward2 += -0.01
            print("Action Correction Reward/Penalty", afrw)
            total_rw = pr + holdings + rwp + afrw
            print("--------------------------------------------------------")
            # Penalty if balance is below minimum threshold
            if action == 0 or action == 1:
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
            bet = end_bet
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
            trade_bias = abs(bc_per - sc_per)
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
            if action == 0 or action == 1:
              print(f"Risk Ratio is: {risk_ratio}")
            else:
              print(f"Risk Ratio is: {start_risk_ratio}")
            print(f"Risk Ratio Factor: {risk_penalty}")
            print(f"Overall Equity Risk Ratio Factor: {pos_rw}")
            print("--------------------------------------------------------")
            if hf != 0:
                print("Total Step Holding/Period Factor:", hf)
                if len(self.positions)==0:
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
                  if len(self.positions)==0:
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
                  if len(self.positions)==0:
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
            
            hold_penalty = 0
            if action == 4 and len(self.positions) > 0:
                self.hold_count += 1

                # Scale based on portfolio baseline (like other rewards)
                scaled_penalty = (0.001 * self.hold_count) * (self.portfolio['total_balance'] / 10000)

                hold_penalty = -scaled_penalty

                if agent_switch == 1:
                    reward += hold_penalty
                else:
                    reward2 += hold_penalty

                total_rw += hold_penalty  # Track in total_rw like other components
                print("Hold penalty with position open is: ", hold_penalty)
            else:
                self.hold_count = 0
                
            fault = False
            if self.portfolio['total_equity'] > (10000 * 0.90) or self.portfolio['total_balance'] > (10000 * 0.90):
                scaled_step = 0.0001 * self.current_step 
                if agent_switch == 1:
                  reward += scaled_step
                  total_rw += scaled_step
                else:
                  reward2 += scaled_step
                  total_rw += scaled_step
                print(f"scaled step reward {scaled_step}")
            #if fault == False:
            #  if agent_switch == 1:
            #    if len(self.positions)==0 and action == 4:
            #      reward += 0
            #    else:
            #      reward += 1
            #  else:
            #    if len(self.positions)==0 and action == 4:
            #      reward2 += 0
            #    else:
            #      reward2 += 1
            
            if self.lifespan == 0 :
                  pacer_eqr = 0
            else:  
                  pacer_eqr = self.avg_check_eqr/self.lifespan
            
            if pacer_eqr < .02/96 and self.lifespan >= 1:
                print(f"Pace is at:  {pacer_eqr}")
                if agent_switch == 1:
                    reward += pacer_eqr * self.lifespan
                    total_rw += pacer_eqr * self.lifespan
                else:
                    reward2 += pacer_eqr * self.lifespan
                    total_rw += pacer_eqr * self.lifespan
            elif pacer_eqr > .02/96 and self.lifespan >= 1:
                print(f"Pace is at:  {pacer_eqr}")
                if agent_switch == 1:
                    reward += pacer_eqr * self.lifespan
                    total_rw += pacer_eqr * self.lifespan
                else:
                    reward2 += pacer_eqr * self.lifespan
                    total_rw += pacer_eqr * self.lifespan
            
            
            flip_bonus = 0
            if action in [0, 1]:
                if self.last_open_action is not None:
                    if action != self.last_open_action:
                        if agent_switch == 1:
                            if total_rw > 0:
                                flip_bonus += abs(total_rw) * 0.50
                            else:
                                flip_bonus -= abs(total_rw) * 0.25
                            reward += flip_bonus
                            total_rw += flip_bonus
                        else:
                            if total_rw > 0:
                                flip_bonus += abs(total_rw) * 0.50
                            else:
                                flip_bonus -= abs(total_rw) * 0.25
                            reward2 += flip_bonus
                            total_rw += flip_bonus
                    if action == self.last_open_action:
                        if agent_switch == 1:
                            flip_bonus -= abs(total_rw) * 0.125
                            reward += flip_bonus
                            total_rw += flip_bonus
                        else:
                            flip_bonus -= abs(total_rw) * 0.125
                            reward2 += flip_bonus
                            total_rw += flip_bonus
                self.last_open_action = action

            q_file = "q_values_select_log_test.csv"
            if os.path.getsize(q_file) > 0:
                df = pd.read_csv(q_file, header=None)
                last_q_row = ast.literal_eval(df.iloc[-1, 0])
                q0, q1, q2, q3 = last_q_row  # or however many actions you have
            else: 
                q0, q1, q2, q3 = 0,0,0,0
            threshold = 0.1
            if agent_switch == 1:
                if len(self.positions) == 0:
                   # Entry situation (0, 1, hold w/ no position)
                   spread = max(q0, q1) - min(q0, q1)
                   if spread < threshold:
                       if total_rw > 0:
                           reward -= abs(total_rw) * 0.5
                           total_rw -= abs(total_rw) * 0.5
                else:
                   # Position is open ??? close vs hold
                   spread = max(q2, q3) - min(q2, q3)
                   if spread < threshold:
                       if total_rw > 0:
                           reward -= abs(total_rw) * 0.5
                           total_rw -= abs(total_rw) * 0.5
            else:
                if len(self.positions) == 0:
                   # Entry situation (0, 1, hold w/ no position)
                   spread = max(q0, q1) - min(q0, q1)
                   if spread < threshold:
                       if total_rw > 0:
                           reward2 -= abs(total_rw) * 0.5
                           total_rw -= abs(total_rw) * 0.5
                else:
                   # Position is open ??? close vs hold
                   spread = max(q2, q3) - min(q2, q3)
                   if spread < threshold:
                       if total_rw > 0:
                           reward2 -= abs(total_rw) * 0.5
                           total_rw -= abs(total_rw) * 0.5
            
            if len(self.positions) == 0 and action == 5:
                if self.portfolio['vault'] > 0:
                    vault_rw = self.portfolio['vault']/self.portfolio['total_balance']
                    if agent_switch == 1:
                        reward += vault_rw 
                        total_rw += vault_rw
                    else:
                        reward2 += vault_rw 
                        total_rw += vault_rw
                        
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
            print("Total Vault:", self.portfolio['vault'])
            print("Total Free Margin:", self.portfolio['total_free_margin'])
            print("Total Equity:", self.portfolio['total_equity'])
            print("Total Buy Options:", self.portfolio['total_buy_options'])
            print("Total Profit from Open Buy Options:", total_buy_profit)
            print("Total Sell Options:", self.portfolio['total_sell_options'])
            print("Total Profit from Open Sell Options:", total_sell_profit)
            print("Total Profit from All Open Options:", total_open_positions_profit)
            
            if action == 5 and check_eqr <= 0:
                self.portfolio['recent_losses'] += 1
            if action == 5 and check_eqr > 0:
                if self.portfolio['recent_losses'] >= 1:
                    self.portfolio['recent_losses'] = 0
                    
            print("consecutive losses are: ", self.portfolio['recent_losses'])
                    
            total_bal = self.portfolio['total_balance']
            # Update current step
            self.current_step += 1
            self.prev_eqr = eqr
            if agent_switch == 1:
                base_step = 0.01
                log_scale = math.log(self.current_step + 2)
                if total_rw < 0:
                    dqn_agent.epsilon += base_step * log_scale / 10  # increase exploration gradually
                else:
                    dqn_agent.epsilon -= base_step * 1 / log_scale  # decrease exploration more early on
                dqn_agent.epsilon = max(min(dqn_agent.epsilon, 0.05), 0.01)
                if dqn_agent.epsilon == .05:
                    agent_switch = 2
                    dqn_agent.epsilon = 0.01
            else:
                base_step = 0.01
                log_scale = math.log(self.current_step + 2)
                if total_rw < 0:
                    dqn_agent.epsilon2 += base_step * log_scale / 10  # increase exploration gradually
                else:
                    dqn_agent.epsilon2 -= base_step * 1 / log_scale  # decrease exploration more early on
                dqn_agent.epsilon2 = max(min(dqn_agent.epsilon2, 0.05), 0.01)
                if dqn_agent.epsilon == .05:
                    agent_switch = 1
                    dqn_agent.epsilon2 = 0.01
            # Determine if the episode is done (you need to define your own termination conditions)
            done = False

            # Episode ends when reaching data boundary
            if self.current_step >= len(self.data) - 1:
                done = True

            # Episode ends if account balance drops too low
            if self.portfolio['total_balance'] < 10000 * 0.9 or self.portfolio['total_equity'] < (10000 * 0.90):
                done = True
                
            return next_state, reward, reward2, total_bal, original_action, action_factor, agent_switch, action, total_rw, done
    
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
           shared_layer1 = Dense(128, activation='relu')(input_layer)
           shared_layer2 = Dense(256, activation='relu')(shared_layer1)
           reshaped_state = Reshape((1, 256))(shared_layer2)
           # Transformer Layer
           num_transformer_layers = 1
           transformer_output = reshaped_state
           for _ in range(num_transformer_layers):
             transformer_output = MultiHeadAttention(num_heads=32, key_dim=self.state_size // 64)(transformer_output, transformer_output)
             transformer_output = LayerNormalization()(transformer_output)
             transformer_output = Dropout(0.1)(transformer_output)
           transformer_output = Flatten()(transformer_output)
           
           num_transformer_layers = 2
           transformer_output2 = reshaped_state
           for _ in range(num_transformer_layers):
             transformer_output2 = MultiHeadAttention(num_heads=32, key_dim=self.state_size // 64)(transformer_output2, transformer_output2)
             transformer_output2 = LayerNormalization()(transformer_output2)
             transformer_output2 = Dropout(0.1)(transformer_output2)
           transformer_output2 = Flatten()(transformer_output2)
             
           num_transformer_layers = 3
           transformer_output3 = reshaped_state
           for _ in range(num_transformer_layers):
             transformer_output3 = MultiHeadAttention(num_heads=32, key_dim=self.state_size // 64)(transformer_output3, transformer_output3)
             transformer_output3 = LayerNormalization()(transformer_output3)
             transformer_output3 = Dropout(0.1)(transformer_output3)
           transformer_output3 = Flatten()(transformer_output3)
             
           num_transformer_layers = 1
           transformer_output4 = reshaped_state
           for _ in range(num_transformer_layers):
             transformer_output4 = MultiHeadAttention(num_heads=32, key_dim=self.state_size // 64)(transformer_output4, transformer_output4)
             transformer_output4 = LayerNormalization()(transformer_output4)
             transformer_output4 = Dropout(0.1)(transformer_output4)
           transformer_output4 = Flatten()(transformer_output4)
             
           num_transformer_layers = 2
           transformer_output5 = reshaped_state
           for _ in range(num_transformer_layers):
             transformer_output5 = MultiHeadAttention(num_heads=32, key_dim=self.state_size // 64)(transformer_output5, transformer_output5)
             transformer_output5 = LayerNormalization()(transformer_output5)
             transformer_output5 = Dropout(0.1)(transformer_output5)
           transformer_output5 = Flatten()(transformer_output5)
             
           num_transformer_layers = 3
           transformer_output6 = reshaped_state
           for _ in range(num_transformer_layers):
             transformer_output6 = MultiHeadAttention(num_heads=32, key_dim=self.state_size // 64)(transformer_output6, transformer_output6)
             transformer_output6 = LayerNormalization()(transformer_output6)
             transformer_output6 = Dropout(0.1)(transformer_output6)
           transformer_output6 = Flatten()(transformer_output6)
           # Value Streams
           value_stream1 = Dense(256, activation='relu')(transformer_output)
           value_stream1 = LayerNormalization()(value_stream1)
           value1 = Dense(1, activation='linear')(value_stream1)

           value_stream2 = Dense(256, activation='relu')(transformer_output2)
           value_stream2 = LayerNormalization()(value_stream2)
           value2 = Dense(1, activation='linear')(value_stream2)
           
           value_stream3 = Dense(256, activation='relu')(transformer_output3)
           value_stream3 = LayerNormalization()(value_stream3)
           value3 = Dense(1, activation='linear')(value_stream3)
           
           value_stream4 = Dense(256, activation='relu')(transformer_output4)
           value_stream4 = LayerNormalization()(value_stream4)
           value4 = Dense(1, activation='linear')(value_stream4)
           
           value_stream5 = Dense(256, activation='relu')(transformer_output5)
           value_stream5 = LayerNormalization()(value_stream5)
           value5 = Dense(1, activation='linear')(value_stream5)
           
           value_stream6 = Dense(256, activation='relu')(transformer_output6)
           value_stream6 = LayerNormalization()(value_stream6)
           value6 = Dense(1, activation='linear')(value_stream6)

           # Advantage Streams
           advantage_stream1 = Dense(256, activation='relu')(transformer_output)
           advantage_stream1 = LayerNormalization()(advantage_stream1)
           advantage1 = Dense(self.action_size, activation='linear')(advantage_stream1)

           advantage_stream2 = Dense(256, activation='relu')(transformer_output2)
           advantage_stream2 = LayerNormalization()(advantage_stream2)
           advantage2 = Dense(self.action_size, activation='linear')(advantage_stream2)
           
           advantage_stream3 = Dense(256, activation='relu')(transformer_output3)
           advantage_stream3 = LayerNormalization()(advantage_stream3)
           advantage3 = Dense(self.action_size, activation='linear')(advantage_stream3)
           
           advantage_stream4 = Dense(256, activation='relu')(transformer_output4)
           advantage_stream4 = LayerNormalization()(advantage_stream4)
           advantage4 = Dense(self.action_size, activation='linear')(advantage_stream4)
           
           advantage_stream5 = Dense(256, activation='relu')(transformer_output5)
           advantage_stream5 = LayerNormalization()(advantage_stream5)
           advantage5 = Dense(self.action_size, activation='linear')(advantage_stream5)
           
           advantage_stream6 = Dense(256, activation='relu')(transformer_output6)
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

        def remember(self, state, original_action, action, reward, next_state, done):
            if agent_switch == 1:
              timestamp = int(datetime.utcnow().strftime('%Y%m%d%H%M%S'))  # e.g., 20250501124556
              self.memory.append((state, original_action, action, reward, next_state, timestamp, done))
            else:
              timestamp = int(datetime.utcnow().strftime('%Y%m%d%H%M%S'))  # e.g., 20250501124556
              self.memory2.append((state, original_action, action, reward, next_state, timestamp, done))

        def act(self, state):
            action_space_mapping = [0, 1, 4, 5]
            q_values = self.model.predict(state)
            q_values2 = self.model2.predict(state)
            
            q_val_list.append(q_values.tolist())  # convert to plain Python list before appending
            with open("q_values_log_test.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(q_values.tolist()) 
                    
            q_val_list2.append(q_values2.tolist())  # convert to plain Python list before appending
            with open("q_values2_log_test.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(q_values2.tolist()) 
                    
            conf1 = np.max(q_values) - np.mean(q_values)
            conf2 = np.max(q_values2) - np.mean(q_values2)
            if conf1 > conf2:
                q_values = q_values
                agent_switch = 1
            else:
                q_values = q_values2
                agent_switch = 2
            if abs(conf1 - conf2) < 0.05:
                q_values = (q_values + q_values2) / 2
                if agent_switch == 1:
                  agent_switch = 2
                else:
                  agent_switch = 1
            
            q_val_list_select.append(q_values.tolist())  # convert to plain Python list before appending
            with open("q_values_select_log_test.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(q_values.tolist()) 
                    
            if agent_switch == 1:
              # Exploration: choose a random action if epsilon-greedy exploration
              if np.random.rand() <= self.epsilon:
                print("Random choice...")
                # Get a list of suitable actions based on open positions
                suitable_actions = []
            
                if not self.env.positions:# and trend_direction == 0:  # If there are no open positions
                    suitable_actions.extend([0,1])  # Choose from actions 0, 1, or 4
                else:
                    if any(position['type'] == 'buy' for position in self.env.positions):# and trend_direction == 0:
                        suitable_actions.extend([4, 5])  # Add actions 2 and 5 if there are open buy positions
                    if any(position['type'] == 'sell' for position in self.env.positions):# and trend_direction == 0:
                        suitable_actions.extend([4, 5])  # Add actions 3 and 5 if there are open sell positions             
                    if any(position['type'] == 'buy' for position in self.env.positions) and any(position['type'] == 'sell' for position in self.env.positions):# and trend_direction == 0:
                        suitable_actions.extend([4, 5])
                suitable_actions = [action for action in suitable_actions if action in action_space_mapping]
                # Map suitable actions to their corresponding model indices
                model_indices = [action_space_mapping.index(action) for action in suitable_actions]
        
                # Filter Q-values to keep only those corresponding to suitable actions
                filtered_q_values = [q_values[0][index] for index in model_indices]
                # **Convert Q-values to probabilities using Softmax**
                softmax_input = np.array(filtered_q_values)
                softmax_probs = scipy.special.softmax(softmax_input)
    
                #action_probabilities = tf.nn.softmax(filtered_q_values).numpy()
                
                # --- Sample an action stochastically based on softmax probabilities ---
                sampled_index = np.random.choice(len(model_indices), p=softmax_probs)
                sampled_action = action_space_mapping[model_indices[sampled_index]]
                sampled_prob = softmax_probs[sampled_index]
    
                # **Get the best action and its probability**
                best_action_index = np.argmax(filtered_q_values)
                best_action = action_space_mapping[model_indices[best_action_index]]
                best_action_probability = softmax_probs[best_action_index]
                return np.random.choice(suitable_actions), best_action_probability
            else:
              # Exploration: choose a random action if epsilon-greedy exploration
              if np.random.rand() <= self.epsilon2:
                print("Random choice...")
                # Get a list of suitable actions based on open positions
                suitable_actions = []
            
                if not self.env.positions:# and trend_direction == 0:  # If there are no open positions
                    suitable_actions.extend([0,1])  # Choose from actions 0, 1, or 4
                else:
                    if any(position['type'] == 'buy' for position in self.env.positions):# and trend_direction == 0:
                        suitable_actions.extend([4, 5])  # Add actions 2 and 5 if there are open buy positions
                    if any(position['type'] == 'sell' for position in self.env.positions):# and trend_direction == 0:
                        suitable_actions.extend([4, 5])  # Add actions 3 and 5 if there are open sell positions             
                    if any(position['type'] == 'buy' for position in self.env.positions) and any(position['type'] == 'sell' for position in self.env.positions):# and trend_direction == 0:
                        suitable_actions.extend([4, 5])
                suitable_actions = [action for action in suitable_actions if action in action_space_mapping]
                # Map suitable actions to their corresponding model indices
                model_indices = [action_space_mapping.index(action) for action in suitable_actions]
        
                # Filter Q-values to keep only those corresponding to suitable actions
                filtered_q_values = [q_values[0][index] for index in model_indices]
                # **Convert Q-values to probabilities using Softmax**
                softmax_input = np.array(filtered_q_values)
                softmax_probs = scipy.special.softmax(softmax_input)
    
                #action_probabilities = tf.nn.softmax(filtered_q_values).numpy()
                
                # --- Sample an action stochastically based on softmax probabilities ---
                sampled_index = np.random.choice(len(model_indices), p=softmax_probs)
                sampled_action = action_space_mapping[model_indices[sampled_index]]
                sampled_prob = softmax_probs[sampled_index]
    
                # **Get the best action and its probability**
                best_action_index = np.argmax(filtered_q_values)
                best_action = action_space_mapping[model_indices[best_action_index]]
                best_action_probability = softmax_probs[best_action_index]
                return np.random.choice(suitable_actions), best_action_probability
            print("Model choice...")
            # Exploitation: choose the action with the highest predicted reward
            # Get suitable actions based on open positions
            suitable_actions = []
            if not self.env.positions:# and trend_direction == 0:  # If there are no open positions
                suitable_actions.extend([0,1])  # Choose from actions 0, 1, or 4
            else:
                if any(position['type'] == 'buy' for position in self.env.positions):# and trend_direction == 0:
                    suitable_actions.extend([4, 5])  # Add actions 2 and 5 if there are open buy positions
                if any(position['type'] == 'sell' for position in self.env.positions):# and trend_direction == 0:
                    suitable_actions.extend([4, 5])  # Add actions 3 and 5 if there are open sell positions
                if any(position['type'] == 'buy' for position in self.env.positions) and any(position['type'] == 'sell' for position in self.env.positions):# and trend_direction == 0:
                    suitable_actions.extend([4, 5])
        
            # Filter suitable actions to ensure they are within the mapped space
            suitable_actions = [action for action in suitable_actions if action in action_space_mapping]

            # Map suitable actions to their corresponding model indices
            model_indices = [action_space_mapping.index(action) for action in suitable_actions]
    
            # Filter Q-values to keep only those corresponding to suitable actions
            filtered_q_values = [q_values[0][index] for index in model_indices]
            # **Convert Q-values to probabilities using Softmax**
            softmax_input = np.array(filtered_q_values)
            softmax_probs = scipy.special.softmax(softmax_input)

            #action_probabilities = tf.nn.softmax(filtered_q_values).numpy()
            
            # --- Sample an action stochastically based on softmax probabilities ---
            sampled_index = np.random.choice(len(model_indices), p=softmax_probs)
            sampled_action = action_space_mapping[model_indices[sampled_index]]
            sampled_prob = softmax_probs[sampled_index]

            # **Get the best action and its probability**
            best_action_index = np.argmax(filtered_q_values)
            best_action = action_space_mapping[model_indices[best_action_index]]
            best_action_probability = softmax_probs[best_action_index]

            print("Filtered Q-Values are: ", filtered_q_values)
            #best_action_index = np.argmax(filtered_q_values)
            #best_action = action_space_mapping[model_indices[best_action_index]]
            #best_action_probability = action_probabilities[best_action_index]

            # **Filtering based on probability threshold ONLY for new trades**
            probability_threshold = 0.7
            print(f"Action {best_action} is @ probability ({best_action_probability:.2f}).")
            print(f"Action {sampled_action} is @ probability ({sampled_prob:.2f}).")
            if not self.env.positions and sampled_action in [0, 1]:  # Only apply if no open positions and action is 0 or 1
                if best_action_probability < probability_threshold:
                    print(f"Action {best_action} rejected due to low probability. Switching to action Sampled Action.")
                    return sampled_action, sampled_prob   # Default to action 4 if confidence is too low

            # **Return the selected action**
            return best_action, best_action_probability

    model_filename = 'agent1.h5'  # Define the model file name for saving and loading your trained model
    model_filename2 = 'agent2.h5'  # Define the model file name for saving and loading your trained model
    
    env = TradingEnvironment(data)  # Initialize your custom trading environment with appropriate data
    env.load_portfolio_normalization_config("portfolio_norm_config.json")
    state_size = len(env.get_state())  # Adjust state size based on your environment's state representation
    action_size = 4  # Set the number of actions according to your trading actions (buy, sell, hold, etc.)
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
        done_loop = False
        failure_done = False
        while not done_loop:
            if env.portfolio['total_equity'] < (10000*0.90) or env.portfolio['total_balance'] < (10000* 0.90):  # Define a custom method in your environment to determine if the episode is done
                done_loop = True
                failure_done = True
                epi = 0.50
                epi2 = 0.50
                dqn_agent.epsilon = epi
                dqn_agent.epsilon2 = epi2
                save_epi(epi, epi_filename)
                save_epi2(epi2, epi_filename2)
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
                action, action_prob = dqn_agent.act(state)
                next_state, reward, reward2, total_bal, original_action, action_factor, agent_switch, action, total_rw, done = env.step(action, reward, reward2, agent_switch, action_prob)  # Implement the step method in your environment
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
                done_loop = True
                
        print(f"The Epsilon is {dqn_agent.epsilon}.")
        print(f"The Epsilon2 is {dqn_agent.epsilon2}.")
        # Visualize episode rewards
        episode_rewards.append(episode_reward+episode_reward2)
        episode_balances.append(episode_balance)
        episode_rewards_df = episode_rewards_df.append({'Episode': episode_count, 'Reward': reward+reward2, 'Balance': env.portfolio['total_balance']}, ignore_index=True)
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
        #if action == 2 or action == 3 or action == 5:
        #  if len(env.positions) < 1:
        #    if reward >= reward2:
        #      agent_switch = 1
        #    else:
        #      agent_switch = 2
        if int(episode_count) == int(total_episodes):
          if env.portfolio['total_balance'] <= int((10000+(10000*.10))):
                failure_done = True
                env.reset() 
                epi = 0.50
                epi2 = 0.50
                save_epi(epi, epi_filename)
                save_epi2(epi2, epi_filename2)
                dqn_agent.epsilon = epi
                dqn_agent.epsilon2 = epi2
        if env.portfolio['total_balance'] >= int((10000+(10000*.10))):
            train = False
        if failure_done == True:
            done = True
            break
    return train, done  
            
dataset = pd.read_csv(r'data\itd_2010-2024.csv')
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
        
if os.path.exists(epi_filename):
  epi = load_epi(epi_filename)
else:
  epi = 0.5
print(f"Loaded Epi is {epi}")

if os.path.exists(epi_filename2):
  epi2 = load_epi2(epi_filename2)
else:
  epi2 = 0.5
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

save_epi(epi, epi_filename)
save_rw(reward, rw_filename)
save_epi2(epi2, epi_filename2)
save_rw2(reward2, rw_filename2)
save_sell_count(sell_count, sc_filename)
save_buy_count(buy_count, bc_filename)
save_reset(reset1, reset1_filename)
save_reset(reset2, reset2_filename)
save_eqrs(eqrs, eqrs_status_filename)

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
if reward <= reward2:
  if reset_p1 > reset_p2 and reset_bias > reset_threshold:
     agent_switch = 2
  else:
    agent_switch = 1
else:
  if reset_p2 > reset_p1 and reset_bias > reset_threshold:
    agent_switch = 1
  else:
    agent_switch = 2

while train == True:
    # Use an absolute file path
    # Remove unnecessary columns (open, high, low, vol, spread)
    data = dataset[['Date', 'Time', 'TickVol', 'High', 'Low', 'Close']]

    # Parse Date column and set it as the index
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data.set_index('Datetime', inplace=True)
    data.drop(columns=['Date'], inplace=True)  # Drop Date column after creating Datetime index

    moving_averages = [48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528]
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
    
    rsi_windows = [48, 96, 144, 192, 240, 288, 336, 384, 432, 480]
    data['RSI4'] = calculate_rsi(data, window=rsi_windows[0])
    data['RSI8'] = calculate_rsi(data, window=rsi_windows[1])
    data['RSI16'] = calculate_rsi(data, window=rsi_windows[2])
    data['RSI32'] = calculate_rsi(data, window=rsi_windows[3])
    data['RSI48'] = calculate_rsi(data, window=rsi_windows[4])
    data['RSI96'] = calculate_rsi(data, window=rsi_windows[5])
    data['RSI192'] = calculate_rsi(data, window=rsi_windows[6])
    data['RSI288'] = calculate_rsi(data, window=rsi_windows[7])
    data['RSI384'] = calculate_rsi(data, window=rsi_windows[8])
    data['RSI480'] = calculate_rsi(data, window=rsi_windows[9])

    # Calculate MACD
    short_window = 48
    long_window = 96
    signal_window = 32
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    data['MACD'] = data['Short_MA'] - data['Long_MA']
    data['Signal_Line'] = data['MACD'].rolling(window=signal_window).mean()
    
    # Calculate Bollinger Bands
    window = 48
    data['Rolling_Mean4'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band4'] = data['Rolling_Mean4'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band4'] = data['Rolling_Mean4'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference4'] = data['Upper_Band4'] - data['Lower_Band4']
    
    # Calculate Bollinger Bands
    window = 96
    data['Rolling_Mean8'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band8'] = data['Rolling_Mean8'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band8'] = data['Rolling_Mean8'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference8'] = data['Upper_Band8'] - data['Lower_Band8']
    
    # Calculate Bollinger Bands
    window = 144
    data['Rolling_Mean16'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band16'] = data['Rolling_Mean16'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band16'] = data['Rolling_Mean16'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference16'] = data['Upper_Band16'] - data['Lower_Band16']
    
    # Calculate Bollinger Bands
    window = 192
    data['Rolling_Mean32'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band32'] = data['Rolling_Mean32'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band32'] = data['Rolling_Mean32'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference32'] = data['Upper_Band32'] - data['Lower_Band32']
    
    # Calculate Bollinger Bands
    window = 240
    data['Rolling_Mean48'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band48'] = data['Rolling_Mean48'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band48'] = data['Rolling_Mean48'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference48'] = data['Upper_Band48'] - data['Lower_Band48']
    
    # Calculate Bollinger Bands
    window = 288
    data['Rolling_Mean96'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band96'] = data['Rolling_Mean96'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band96'] = data['Rolling_Mean96'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference96'] = data['Upper_Band96'] - data['Lower_Band96']
    
    # Calculate Bollinger Bands
    window = 336
    data['Rolling_Mean192'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band192'] = data['Rolling_Mean192'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band192'] = data['Rolling_Mean192'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference192'] = data['Upper_Band192'] - data['Lower_Band192']
    
    # Calculate Bollinger Bands
    window = 384
    data['Rolling_Mean288'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band288'] = data['Rolling_Mean288'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band288'] = data['Rolling_Mean288'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference288'] = data['Upper_Band288'] - data['Lower_Band288']
    
    # Calculate Bollinger Bands
    window = 432
    data['Rolling_Mean384'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band384'] = data['Rolling_Mean384'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band384'] = data['Rolling_Mean384'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference384'] = data['Upper_Band384'] - data['Lower_Band384']
    
    # Calculate Bollinger Bands
    window = 480
    data['Rolling_Mean480'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band480'] = data['Rolling_Mean480'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band480'] = data['Rolling_Mean480'] - 2 * data['Close'].rolling(window=window).std()
    data['Band_Difference480'] = data['Upper_Band480'] - data['Lower_Band480']

    # Calculate Stochastic Oscillator
    k_window = 96
    d_window = 32
    data['Lowest_Low'] = data['Low'].rolling(window=k_window).min()
    data['Highest_High'] = data['High'].rolling(window=k_window).max()
    data['%K'] = (data['Close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low']) * 100
    data['%D'] = data['%K'].rolling(window=d_window).mean()

    # Calculate Price Rate of Change (ROC)
    roc_window = 48
    data['ROC4'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100
    roc_window = 96
    data['ROC8'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100
    roc_window = 144
    data['ROC16'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100
    roc_window = 192
    data['ROC32'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100
    roc_window = 240
    data['ROC48'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100
    roc_window = 288
    data['ROC96'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100
    roc_window = 336
    data['ROC192'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100
    roc_window = 384
    data['ROC288'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100
    roc_window = 432
    data['ROC384'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100
    roc_window = 480
    data['ROC480'] = (data['Close'] - data['Close'].shift(roc_window)) / data['Close'].shift(roc_window) * 100

    # Calculate Average True Range (ATR)
    atr_window = 48
    data['TR4'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR4'] = data['TR4'].rolling(window=atr_window).mean()
    atr_window = 96
    data['TR8'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR8'] = data['TR8'].rolling(window=atr_window).mean()
    atr_window = 144
    data['TR16'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR16'] = data['TR16'].rolling(window=atr_window).mean()
    atr_window = 192
    data['TR32'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR32'] = data['TR32'].rolling(window=atr_window).mean()
    atr_window = 240
    data['TR48'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR48'] = data['TR48'].rolling(window=atr_window).mean()
    atr_window = 288
    data['TR96'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR96'] = data['TR96'].rolling(window=atr_window).mean()
    atr_window = 336
    data['TR192'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR192'] = data['TR192'].rolling(window=atr_window).mean()
    atr_window = 384
    data['TR288'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR288'] = data['TR288'].rolling(window=atr_window).mean()
    atr_window = 432
    data['TR384'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR384'] = data['TR384'].rolling(window=atr_window).mean()
    atr_window = 480
    data['TR480'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    data['ATR480'] = data['TR480'].rolling(window=atr_window).mean()
    
    data['Pip_Range4'] = (data['Band_Difference4'] + data['ATR4']) / 2
    data['Smoothed_Pip_Range4'] = data['Pip_Range4'].rolling(window=48).mean()
    
    data['Pip_Range8'] = (data['Band_Difference8'] + data['ATR8']) / 2
    data['Smoothed_Pip_Range8'] = data['Pip_Range8'].rolling(window=96).mean()
    
    data['Pip_Range16'] = (data['Band_Difference16'] + data['ATR16']) / 2
    data['Smoothed_Pip_Range16'] = data['Pip_Range16'].rolling(window=144).mean()
    
    data['Pip_Range32'] = (data['Band_Difference32'] + data['ATR32']) / 2
    data['Smoothed_Pip_Range32'] = data['Pip_Range32'].rolling(window=192).mean()
    
    data['Pip_Range48'] = (data['Band_Difference48'] + data['ATR48']) / 2
    data['Smoothed_Pip_Range48'] = data['Pip_Range48'].rolling(window=240).mean()
    
    data['Pip_Range96'] = (data['Band_Difference96'] + data['ATR96']) / 2
    data['Smoothed_Pip_Range96'] = data['Pip_Range96'].rolling(window=288).mean()
    
    data['Pip_Range192'] = (data['Band_Difference192'] + data['ATR192']) / 2
    data['Smoothed_Pip_Range192'] = data['Pip_Range192'].rolling(window=336).mean()
    
    data['Pip_Range288'] = (data['Band_Difference288'] + data['ATR288']) / 2
    data['Smoothed_Pip_Range288'] = data['Pip_Range288'].rolling(window=384).mean()
    
    data['Pip_Range384'] = (data['Band_Difference384'] + data['ATR384']) / 2
    data['Smoothed_Pip_Range384'] = data['Pip_Range384'].rolling(window=432).mean()
    
    data['Pip_Range480'] = (data['Band_Difference480'] + data['ATR480']) / 2
    data['Smoothed_Pip_Range480'] = data['Pip_Range480'].rolling(window=480).mean()

    # Calculate On-Balance Volume (OBV)
    data['Volume_Direction'] = data['TickVol'].apply(lambda x: 1 if x >= 0 else -1)
    data['OBV'] = data['Volume_Direction'] * data['TickVol']
    data['OBV'] = data['OBV'].cumsum()
    data['OBV'] = data['OBV'].rolling(window=32).mean()

    # Calculate Average Directional Index (ADX)
    adx_window = 96
    data['High_Low'] = data['High'] - data['Low']
    data['High_Prev_Close'] = abs(data['High'] - data['Close'].shift(1))
    data['Low_Prev_Close'] = abs(data['Low'] - data['Close'].shift(1))
    data['+DM'] = data[['High_Low', 'High_Prev_Close']].apply(lambda x: x['High_Low'] if x['High_Low'] > x['High_Prev_Close'] else 0, axis=1)
    data['-DM'] = data[['High_Low', 'Low_Prev_Close']].apply(lambda x: x['Low_Prev_Close'] if x['Low_Prev_Close'] > x['High_Low'] else 0, axis=1)
    data['+DI'] = (data['+DM'].rolling(window=adx_window).mean() / data['ATR96'].rolling(window=adx_window).mean()) * 100
    data['-DI'] = (data['-DM'].rolling(window=adx_window).mean() / data['ATR96'].rolling(window=adx_window).mean()) * 100
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
    compute_and_save_market_stats()

    class TradingEnvironment:
        def __init__(self, data, total_episodes):
            self.data = data
            self.current_step = 0
            self.reset1 = reset1
            self.reset2 = reset2
            self.fee_rate = 0.10
            self.lifespan = 0
            self.prev_eqr = 0
            self.positions = []  # Dictionary to track open buy and sell options
            self.profit_history = []  
            self.positive_count = 0
            self.negative_count = 0
            self.avg_check_eqr_list = []
            self.avg_check_eqr = 0
            self.npo_hp = 0
            self.eqr_switch = False
            self.hold_count = 0
            self.last_boost_step = -10 
            self.last_open_action = None
            self.range_mode = False
            self.market_mean = np.load("market_mean.npy")
            self.market_std = np.load("market_std.npy")
            self.portfolio_keys = [  # must match order in your portfolio_state array
                'total_balance', 'total_free_margin', 'total_equity', 'total_buy_options', 'total_sell_options',
                'sell_count', 'buy_count', 'start_hf', 'hold_fct', 'start_port', 'top_eqr', 'btm_eqr', 'check_bet',
                'check_eqr', 'action_factor', 'start_risk_ratio', 'start_bet', 'hf', 'end_port', 'reward_port',
                'rwp', 'holdings', 'pr', 'total_rw', 'risk_penalty', 'end_bet', 'bet', 'eqr', 'erw',
                'sc_per', 'bc_per', 'count_dist', 'trade_bias', 'pos_rw', 'reward', 'reward2', 'lifespan',
                'percent_positive', 'percent_negative', 'target_tier', 'avg_check_eqr', 'recent_losses', 'vault'
            ]
            self.categorical_portfolio_keys = {'target_tier', 'action_factor', 'trade_bias'}
            self.portfolio_mins = np.full(len(self.portfolio_keys), np.inf)
            self.portfolio_maxs = np.full(len(self.portfolio_keys), -np.inf)
            self.portfolio_normalization_config = None
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
                'top_eqr': 0.05,
                'btm_eqr': -0.05,
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
                'lifespan': 0,
                'percent_positive': 0,
                'percent_negative': 0,
                'target_tier': 0,
                'avg_check_eqr': 0,
                'recent_losses': 0,
                'vault': 0
                #'avg_check_eqr_list': [],
                #'trailing_stop': 0
            }

        def reset(self):
            self.current_step = 0
            self.positions = []  # Reset positions dictionary
            self.profit_history = [] 
            self.positive_count = 0
            self.negative_count = 0
            self.avg_check_eqr_list = []
            self.avg_check_eqr = 0
            self.npo_hp = 0
            self.eqr_switch = False
            self.hold_count = 0
            self.last_open_action = None
            self.range_mode = False
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
                'top_eqr': 0.05,
                'btm_eqr': -0.05,
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
                'lifespan': 0,
                'percent_positive': 0,
                'percent_negative': 0,
                'target_tier': 0,
                'avg_check_eqr': 0,
                'recent_losses': 0,
                'vault': 0
                #'avg_check_eqr_list': [],
                #'trailing_stop': 0
            }
        
        def save_portfolio_normalization_config(self, filename="portfolio_norm_config.json", buffer=0.05):
            config = {}

            for i, key in enumerate(self.portfolio_keys):
                low = float(self.portfolio_mins[i]) * (1 - buffer)
                high = float(self.portfolio_maxs[i]) * (1 + buffer)
                config[key] = [low, high]

            with open(filename, "w") as f:
                json.dump(config, f, indent=2)

            print(f"Saved portfolio normalization config to {filename}")
        
        def load_portfolio_normalization_config(self, filename="portfolio_norm_config.json"):
            with open(filename, "r") as f:
                self.portfolio_normalization_config = json.load(f)

            print(f"Loaded portfolio normalization config from {filename}")
            
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
                current_data['MA_7'],
                current_data['MA_8'],
                current_data['MA_9'],
                current_data['MA_10'],
                current_data['MA_11'],
                current_data['RSI4'],
                current_data['RSI8'],
                current_data['RSI16'],
                current_data['RSI32'],
                current_data['RSI48'],
                current_data['RSI96'],
                current_data['RSI192'],
                current_data['RSI288'],
                current_data['RSI384'],
                current_data['RSI480'],
                current_data['Short_MA'],
                current_data['Long_MA'],
                current_data['MACD'],
                current_data['Signal_Line'],
                current_data['Rolling_Mean4'],
                current_data['Upper_Band4'],
                current_data['Lower_Band4'],
                current_data['Band_Difference4'],
                current_data['Rolling_Mean8'],
                current_data['Upper_Band8'],
                current_data['Lower_Band8'],
                current_data['Band_Difference8'],
                current_data['Rolling_Mean16'],
                current_data['Upper_Band16'],
                current_data['Lower_Band16'],
                current_data['Band_Difference16'],
                current_data['Rolling_Mean32'],
                current_data['Upper_Band32'],
                current_data['Lower_Band32'],
                current_data['Band_Difference32'],
                current_data['Rolling_Mean48'],
                current_data['Upper_Band48'],
                current_data['Lower_Band48'],
                current_data['Band_Difference48'],
                current_data['Rolling_Mean96'],
                current_data['Upper_Band96'],
                current_data['Lower_Band96'],
                current_data['Band_Difference96'],
                current_data['Rolling_Mean192'],
                current_data['Upper_Band192'],
                current_data['Lower_Band192'],
                current_data['Band_Difference192'],
                current_data['Rolling_Mean288'],
                current_data['Upper_Band288'],
                current_data['Lower_Band288'],
                current_data['Band_Difference288'],
                current_data['Rolling_Mean384'],
                current_data['Upper_Band384'],
                current_data['Lower_Band384'],
                current_data['Band_Difference384'],
                current_data['Rolling_Mean480'],
                current_data['Upper_Band480'],
                current_data['Lower_Band480'],
                current_data['Band_Difference480'],
                current_data['Lowest_Low'],
                current_data['Highest_High'],
                current_data['%K'],
                current_data['%D'],
                current_data['ROC4'],
                current_data['ROC8'],
                current_data['ROC16'],
                current_data['ROC32'],
                current_data['ROC48'],
                current_data['ROC96'],
                current_data['ROC192'],
                current_data['ROC288'],
                current_data['ROC384'],
                current_data['ROC480'],
                current_data['TR4'],
                current_data['ATR4'],
                current_data['TR8'],
                current_data['ATR8'],
                current_data['TR16'],
                current_data['ATR16'],
                current_data['TR32'],
                current_data['ATR32'],
                current_data['TR48'],
                current_data['ATR48'],
                current_data['TR96'],
                current_data['ATR96'],
                current_data['TR192'],
                current_data['ATR192'],
                current_data['TR288'],
                current_data['ATR288'],
                current_data['TR384'],
                current_data['ATR384'],
                current_data['TR480'],
                current_data['ATR480'],
                current_data['Pip_Range4'],
                current_data['Smoothed_Pip_Range4'],
                current_data['Pip_Range8'],
                current_data['Smoothed_Pip_Range8'],
                current_data['Pip_Range16'],
                current_data['Smoothed_Pip_Range16'],
                current_data['Pip_Range32'],
                current_data['Smoothed_Pip_Range32'],
                current_data['Pip_Range48'],
                current_data['Smoothed_Pip_Range48'],
                current_data['Pip_Range96'],
                current_data['Smoothed_Pip_Range96'],
                current_data['Pip_Range192'],
                current_data['Smoothed_Pip_Range192'],
                current_data['Pip_Range288'],
                current_data['Smoothed_Pip_Range288'],
                current_data['Pip_Range384'],
                current_data['Smoothed_Pip_Range384'],
                current_data['Pip_Range480'],
                current_data['Smoothed_Pip_Range480'],
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
            normalized_market = (current_features - self.market_mean) / (self.market_std + 1e-6)
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
                self.portfolio['lifespan'], 
                self.portfolio['percent_positive'],
                self.portfolio['percent_negative'],
                self.portfolio['target_tier'],
                self.portfolio['avg_check_eqr'],
                self.portfolio['recent_losses'],
                self.portfolio['vault']
                #self.portfolio['avg_check_eqr_list'],
                #self.portfolio['trailing_stop']
            ], dtype=np.float32)
            self.portfolio_mins = np.minimum(self.portfolio_mins, portfolio_state)
            self.portfolio_maxs = np.maximum(self.portfolio_maxs, portfolio_state)
            normalized_portfolio = []
            for i, (key, value) in enumerate(zip(self.portfolio_keys, portfolio_state)):
                if key in self.categorical_portfolio_keys:
                    norm_value = float(value)  # Pass through unnormalized (or one-hot outside loop)
                else:
                    if self.portfolio_normalization_config:
                        low, high = self.portfolio_normalization_config[key]
                    else:
                        low, high = self.portfolio_mins[i], self.portfolio_maxs[i]
                    norm_value = (value - low) / (high - low + 1e-6)
                    norm_value = np.clip(norm_value, 0, 1)
                normalized_portfolio.append(norm_value)

            normalized_portfolio = np.array(normalized_portfolio, dtype=np.float32)
            
            # Concatenate current features and portfolio information
            state = np.concatenate((normalized_market, normalized_portfolio))

            return state
    
        def step(self, action, reward, reward2, agent_switch, action_prob):
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
            if os.path.exists(reset1_filename):
              self.reset1 = load_reset1(reset1_filename)
            else:
              self.reset1 = 0
            print(f"Reset1 is {self.reset1}")
            if os.path.exists(reset2_filename):
              self.reset2 = load_reset2(reset2_filename)
            else:
              self.reset2 = 0
            print(f"Reset2 is {self.reset2}")
            start_hf = 0
            max_holding_period = (96*5)-1
            holding_period_factors = []
            for position in self.positions:
                holding_period = position['open_step']
                if holding_period != 0:
                    holding_period_factor = (holding_period / max_holding_period)# Adjust this as needed
                else:
                    holding_period_factor = 0
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
            
            if len(self.positions) == 0: 
                self.profit_history = []
                self.positive_count = 0
                self.negative_count = 0
            else:
                # Store the profit in the history
                self.profit_history.append(total_open_positions_profit)
                # Update positive/negative counts
                if total_open_positions_profit > 0:
                    self.positive_count += 1
                else:
                    self.negative_count += 1
            print(f"Count of steps with profit above 0: {self.positive_count:.2f}")
            print(f"Count of steps with profit below or equal to 0: {self.negative_count:.2f}")
            # Calculate percentages
            if (self.positive_count + self.negative_count) != 0:
                 percent_positive = (self.positive_count / (self.positive_count + self.negative_count))
                 percent_negative = (self.negative_count / (self.positive_count + self.negative_count)) 
            else:
                percent_positive = 0
                percent_negative = 0

            self.portfolio['percent_positive'] = percent_positive
            self.portfolio['negative_positive'] = percent_positive
            # Output the results
            print(f"Percentage of steps with profit above 0: {percent_positive:.2f}%")
            print(f"Percentage of steps with profit below or equal to 0: {percent_negative:.2f}%")
            
            fee = 0
            prev_total_balance = self.portfolio['total_balance']
            
            if action == 0 or action == 1:
                base_pz = self.portfolio['total_free_margin'] / 1000.0
                if self.portfolio['recent_losses'] >= 1: 
                    action_prob *= 0.5
                scaled_confidence = max(0.01, min(action_prob, 1.0))  # clip to [0.01, 1.0]
                adjusted_pz = base_pz * scaled_confidence
                action_pz = min(max(1, adjusted_pz), 10)
            else:
                action_pz = 0
            pz = action_pz #self.portfolio['total_free_margin']/1000
            if pz > 10:
              pz = 10
            elif pz < 1:
                pz = 1
            print("The posiiton size is: ", pz)
            action_factor = 1
            #ATR = self.data['ATR'][self.current_step]
            if len(self.positions) == 0:
              bet_scale = pz * (self.portfolio['total_free_margin']*0.01)
            else:
              bet_scale = (self.portfolio['total_balance']-self.portfolio['total_free_margin'])
            scaled_value4 = ((self.data['Smoothed_Pip_Range4'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value4 < 0.01:
              scaled_value4 = 0.01
            print("The scaled target 4 is ", scaled_value4)
            scaled_value8 = ((self.data['Smoothed_Pip_Range8'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value8 < 0.01:
              scaled_value8 = 0.01
            print("The scaled target 8 is ", scaled_value8)
            scaled_value16 = ((self.data['Smoothed_Pip_Range16'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value16 < 0.01:
              scaled_value16 = 0.01
            print("The scaled target 16 is ", scaled_value16)
            scaled_value32 = ((self.data['Smoothed_Pip_Range32'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value32 < 0.01:
              scaled_value32 = 0.01
            print("The scaled target 32 is ", scaled_value32)
            scaled_value48 = ((self.data['Smoothed_Pip_Range48'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value48 < 0.01:
              scaled_value48 = 0.01
            print("The scaled target 48 is ", scaled_value48)
            scaled_value96 = ((self.data['Smoothed_Pip_Range96'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value96 < 0.01:
              scaled_value96 = 0.01
            print("The scaled target 96 is ", scaled_value96)
            scaled_value192 = ((self.data['Smoothed_Pip_Range192'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value192 < 0.01:
              scaled_value192 = 0.01
            print("The scaled target 192 is ", scaled_value192)
            scaled_value288 = ((self.data['Smoothed_Pip_Range288'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value288 < 0.01:
              scaled_value288 = 0.01
            print("The scaled target 288 is ", scaled_value288)
            scaled_value384 = ((self.data['Smoothed_Pip_Range384'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value384 < 0.01:
              scaled_value384 = 0.01
            print("The scaled target 384 is ", scaled_value384)
            scaled_value480 = ((self.data['Smoothed_Pip_Range480'][self.current_step] * 10000)*pz)/bet_scale
            if scaled_value480 < 0.01:
              scaled_value480 = 0.01
            print("The scaled target 480 is ", scaled_value480)
            #top_eqr = (2)*scaled_value
            #btm_eqr = (-1)*scaled_value
            base_rate = 0.1  # Minimum transition rate (10%)
            volatility_factor4 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))
            volatility_factor8 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))
            volatility_factor16 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))
            volatility_factor32 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))
            volatility_factor48 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))
            volatility_factor96 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))
            volatility_factor192 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))
            volatility_factor288 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))
            volatility_factor384 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))
            volatility_factor480 = max(1.0, min(3.0, 1 + (self.data['Smoothed_Pip_Range4'][self.current_step] * 50)))

            # Calculate dynamic transition rate for each scaled value
            transition_rate4 = min(max(base_rate + (volatility_factor4 * scaled_value4), 0.1), 0.5)
            transition_rate8 = min(max(base_rate + (volatility_factor8 * scaled_value8), 0.1), 0.5)
            transition_rate16 = min(max(base_rate + (volatility_factor16 * scaled_value16), 0.1), 0.5)
            transition_rate32 = min(max(base_rate + (volatility_factor32 * scaled_value32), 0.1), 0.5)
            transition_rate48 = min(max(base_rate + (volatility_factor48 * scaled_value48), 0.1), 0.5)
            transition_rate96 = min(max(base_rate + (volatility_factor96 * scaled_value96), 0.1), 0.5)
            transition_rate192 = min(max(base_rate + (volatility_factor192 * scaled_value192), 0.1), 0.5)
            transition_rate288 = min(max(base_rate + (volatility_factor288 * scaled_value288), 0.1), 0.5)
            transition_rate384 = min(max(base_rate + (volatility_factor384 * scaled_value384), 0.1), 0.5)
            transition_rate480 = min(max(base_rate + (volatility_factor480 * scaled_value480), 0.1), 0.5)

            #top_eqr = (2)*scaled_value
            #btm_eqr = (-1)*scaled_value
            tfmc = self.portfolio['total_free_margin']
            check_bet = (self.portfolio['total_balance']-tfmc)
            if start_port != 0 and check_bet != 0:
              check_eqr = start_port/check_bet
            else:
              check_eqr = 0
              
            if len(self.positions) == 0: 
                self.avg_check_eqr_list = []
                self.avg_check_eqr = 0
            else:
                self.avg_check_eqr_list.append(check_eqr)
                if sum(self.avg_check_eqr_list) == 0 or len(self.avg_check_eqr_list) == 0:
                    self.avg_check_eqr = 0
                else:
                    # Number of values in the list
                    n = len(self.avg_check_eqr_list)
        
                    # Base decay rate and dynamic scaling
                    base_rate = 0.05  # Adjust this as needed
                    decay_rate = base_rate * math.log(n + 1)  # Logarithmic scaling
        
                    # Exponential decay for the newest weight
                    newest_weight = math.exp(-decay_rate * n)

                    # Calculate remaining weight
                    remaining_weight = 1 - newest_weight

                    # Descending proportions for older values
                    descending_proportions = list(range(n - 1, 0, -1))  # Example: [4, 3, 2, 1] for n=5
                    total_proportions = sum(descending_proportions)

                    # Calculate weights for older values
                    additional_weights = [
                        remaining_weight * (p / total_proportions) for p in descending_proportions
                    ]

                    # Combine weights with the newest value
                    weights = additional_weights + [newest_weight]

                    # Reverse weights so the newest (last value) has the highest weight
                    weights.reverse()

                    # Calculate weighted mean
                    weighted_mean = sum(value * weight for value, weight in zip(self.avg_check_eqr_list, weights))
        
                    # Update the average
                    self.avg_check_eqr = weighted_mean
            #self.portfolio['avg_check_eqr_list'] = self.avg_check_eqr_list
            self.portfolio['avg_check_eqr'] = self.avg_check_eqr
            print(f"The average eqr is: {self.avg_check_eqr}")
            
            if len(self.avg_check_eqr_list) >= 1:
                median_eqr = statistics.median([abs(x) for x in self.avg_check_eqr_list])
            else:
                median_eqr = 0
            print(f"The median eqr is: {median_eqr}")
            
            if len(self.positions) == 0:
                top_eqr = scaled_value4
                btm_eqr = (scaled_value4*-1)/2
                if scaled_value4 < 0.20:
                    self.range_mode = True
            else:
                top_eqr = self.portfolio['top_eqr']
                btm_eqr = self.portfolio['btm_eqr']
            self.portfolio['top_eqr'] = top_eqr
            self.portfolio['btm_eqr'] = btm_eqr
            ttb1 = btm_eqr
            ttb2 = btm_eqr
            ttb3 = btm_eqr
            ttb4 = btm_eqr
            ttb5 = btm_eqr
            ttb6 = btm_eqr
            ttb7 = btm_eqr
            ttb8 = btm_eqr
            ttb9 = btm_eqr
            ttb10 = btm_eqr
            if len(self.positions)==0:
                 if self.eqr_switch == True:
                     self.eqr_switch = False
                 if self.range_mode == True and scaled_value4 >= 0.20:
                     self.range_mode = False
            if self.portfolio['lifespan'] % 4 == 0:
                if median_eqr < 0.10 and scaled_value4 < 0.20 and self.portfolio['lifespan'] % 48 == 0 and self.portfolio['lifespan'] != 0:
                    top_eqr = self.avg_check_eqr
                    btm_eqr = btm_eqr
                    self.range_mode = True
                    print("Ranging Mode...")
                else:
                    print("Trending Mode...")
                    if 4 <= self.portfolio['lifespan'] < 32 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate4) * top_eqr + transition_rate4 * scaled_value4
                         btm_eqr = top_eqr / -2
                         ttb1 = btm_eqr
                         self.portfolio['target_tier'] = 1
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
                    elif 32 <= self.portfolio['lifespan'] < 64 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate8) * top_eqr + transition_rate8 * scaled_value8
                         if btm_eqr >= 0:
                             if self.eqr_switch == False:
                                 btm_eqr = min((1 - transition_rate8) * btm_eqr + transition_rate8 * scaled_value8, (top_eqr/2))
                                 print("Positive transition btm_eqr is: ", btm_eqr)
                             else:
                                 print("No Positive transition eqr switch True  btm_eqr is: ", btm_eqr)
                         else:
                             check_btm = (1 - transition_rate8) * btm_eqr + transition_rate8 * scaled_value8
                             if check_eqr < check_btm:
                                 if check_eqr > top_eqr/2:
                                     btm_eqr = btm_eqr/2
                                 else:
                                     btm_eqr = btm_eqr
                             else: 
                                btm_eqr = check_btm
                             print("Negtive transition btm_eqr is: ", btm_eqr)
                         if -0.01 < btm_eqr < 0:
                             btm_eqr = 0
                         ttb2 = btm_eqr
                         self.portfolio['target_tier'] = 2
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
                    elif 64 <= self.portfolio['lifespan'] < 96 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate16) * top_eqr + transition_rate16 * scaled_value16
                         if btm_eqr >= 0:
                             if self.eqr_switch == False:
                                 btm_eqr = min((1 - transition_rate16) * btm_eqr + transition_rate16 * scaled_value16, (top_eqr/2))
                                 print("Positive transition btm_eqr is: ", btm_eqr)
                             else:
                                 print("No Positive transition eqr switch True  btm_eqr is: ", btm_eqr)
                         else:
                             check_btm = (1 - transition_rate16) * btm_eqr + transition_rate16 * scaled_value16
                             if check_eqr < check_btm:
                                 if check_eqr > top_eqr/2:
                                     btm_eqr = btm_eqr/2
                                 else:
                                     btm_eqr = btm_eqr
                             else: 
                                btm_eqr = check_btm
                             print("Negtive transition btm_eqr is: ", btm_eqr)
                         if -0.01 < btm_eqr < 0:
                             btm_eqr = 0
                         ttb3 = btm_eqr
                         self.portfolio['target_tier'] = 3
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
                    elif 96 <= self.portfolio['lifespan'] < 128 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate32) * top_eqr + transition_rate32 * scaled_value32
                         if btm_eqr >= 0:
                             if self.eqr_switch == False:
                                 btm_eqr = min((1 - transition_rate32) * btm_eqr + transition_rate32 * scaled_value32, (top_eqr/2))
                                 print("Positive transition btm_eqr is: ", btm_eqr)
                             else:
                                 print("No Positive transition eqr switch True  btm_eqr is: ", btm_eqr)
                         else:
                             check_btm = (1 - transition_rate32) * btm_eqr + transition_rate32 * scaled_value32
                             if check_eqr < check_btm:
                                 if check_eqr > top_eqr/2:
                                     btm_eqr = btm_eqr/2
                                 else:
                                     btm_eqr = btm_eqr
                             else: 
                                btm_eqr = check_btm
                             print("Negtive transition btm_eqr is: ", btm_eqr)
                         if -0.01 < btm_eqr < 0:
                             btm_eqr = 0
                         ttb4 = btm_eqr
                         self.portfolio['target_tier'] = 4
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
                    elif 128 <= self.portfolio['lifespan'] < 160 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate48) * top_eqr + transition_rate48 * scaled_value48
                         if btm_eqr >= 0:
                             if self.eqr_switch == False:
                                 btm_eqr = min((1 - transition_rate48) * btm_eqr + transition_rate48 * scaled_value48, (top_eqr/2))
                                 print("Positive transition btm_eqr is: ", btm_eqr)
                             else:
                                 print("No Positive transition eqr switch True  btm_eqr is: ", btm_eqr)
                         else:
                             check_btm = (1 - transition_rate48) * btm_eqr + transition_rate48 * scaled_value48
                             if check_eqr < check_btm:
                                 if check_eqr > top_eqr/2:
                                     btm_eqr = btm_eqr/2
                                 else:
                                     btm_eqr = btm_eqr
                             else: 
                                btm_eqr = check_btm
                             print("Negtive transition btm_eqr is: ", btm_eqr)
                         if -0.01 < btm_eqr < 0:
                             btm_eqr = 0
                         ttb5 = btm_eqr
                         self.portfolio['target_tier'] = 5
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
                    elif 160 <= self.portfolio['lifespan'] < 192 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate96) * top_eqr + transition_rate96 * scaled_value96
                         if btm_eqr >= 0:
                             if self.eqr_switch == False:
                                 btm_eqr = min((1 - transition_rate96) * btm_eqr + transition_rate96 * scaled_value96, (top_eqr/2))
                                 print("Positive transition btm_eqr is: ", btm_eqr)
                             else:
                                 print("No Positive transition eqr switch True  btm_eqr is: ", btm_eqr)
                         else:
                             check_btm = (1 - transition_rate96) * btm_eqr + transition_rate96 * scaled_value96
                             if check_eqr < check_btm:
                                 if check_eqr > top_eqr/2:
                                     btm_eqr = btm_eqr/2
                                 else:
                                     btm_eqr = btm_eqr
                             else: 
                                btm_eqr = check_btm
                             print("Negtive transition btm_eqr is: ", btm_eqr)
                         if -0.01 < btm_eqr < 0:
                             btm_eqr = 0
                         ttb6 = btm_eqr
                         self.portfolio['target_tier'] = 6
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
                    elif  192 <= self.portfolio['lifespan'] < 224 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate192) * top_eqr + transition_rate192 * scaled_value192
                         if btm_eqr >= 0:
                             if self.eqr_switch == False:
                                 btm_eqr = min((1 - transition_rate192) * btm_eqr + transition_rate192 * scaled_value192, (top_eqr/2))
                                 print("Positive transition btm_eqr is: ", btm_eqr)
                             else:
                                 print("No Positive transition eqr switch True  btm_eqr is: ", btm_eqr)
                         else:
                             check_btm = (1 - transition_rate192) * btm_eqr + transition_rate192 * scaled_value192
                             if check_eqr < check_btm:
                                 if check_eqr > top_eqr/2:
                                     btm_eqr = btm_eqr/2
                                 else:
                                     btm_eqr = btm_eqr
                             else: 
                                btm_eqr = check_btm
                             print("Negtive transition btm_eqr is: ", btm_eqr)
                         if -0.01 < btm_eqr < 0:
                             btm_eqr = 0
                         ttb7 = btm_eqr
                         self.portfolio['target_tier'] = 7
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
                    elif  224 <= self.portfolio['lifespan'] < 256 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate288) * top_eqr + transition_rate288 * scaled_value288
                         if btm_eqr >= 0:
                             if self.eqr_switch == False:
                                 btm_eqr = min((1 - transition_rate288) * btm_eqr + transition_rate288 * scaled_value288, (top_eqr/2))
                                 print("Positive transition btm_eqr is: ", btm_eqr)
                             else:
                                 print("No Positive transition eqr switch True  btm_eqr is: ", btm_eqr)
                         else:
                             check_btm = (1 - transition_rate288) * btm_eqr + transition_rate288 * scaled_value288
                             if check_eqr < check_btm:
                                 if check_eqr > top_eqr/2:
                                     btm_eqr = btm_eqr/2
                                 else:
                                     btm_eqr = btm_eqr
                             else: 
                                btm_eqr = check_btm
                             print("Negtive transition btm_eqr is: ", btm_eqr)
                         if -0.01 < btm_eqr < 0:
                             btm_eqr = 0
                         ttb8 = btm_eqr
                         self.portfolio['target_tier'] = 8
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
                    elif  256 <= self.portfolio['lifespan'] < 288 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate384) * top_eqr + transition_rate384 * scaled_value384
                         if btm_eqr >= 0:
                             if self.eqr_switch == False:
                                 btm_eqr = min((1 - transition_rate384) * btm_eqr + transition_rate384 * scaled_value384, (top_eqr/2))
                                 print("Positive transition btm_eqr is: ", btm_eqr)
                             else:
                                 print("No Positive transition eqr switch True  btm_eqr is: ", btm_eqr)
                         else:
                             check_btm = (1 - transition_rate384) * btm_eqr + transition_rate384 * scaled_value384
                             if check_eqr < check_btm:
                                 if check_eqr > top_eqr/2:
                                     btm_eqr = btm_eqr/2
                                 else:
                                     btm_eqr = btm_eqr
                             else: 
                                btm_eqr = check_btm
                             print("Negtive transition btm_eqr is: ", btm_eqr)
                         if -0.01 < btm_eqr < 0:
                             btm_eqr = 0
                         ttb9 = btm_eqr
                         self.portfolio['target_tier'] = 9
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
                    elif  288 <= self.portfolio['lifespan'] < 320 and percent_positive >= 0.66 and self.avg_check_eqr > (btm_eqr/2):
                         top_eqr = (1 - transition_rate480) * top_eqr + transition_rate480 * scaled_value480
                         if btm_eqr >= 0:
                             if self.eqr_switch == False:
                                 btm_eqr = min((1 - transition_rate480) * btm_eqr + transition_rate480 * scaled_value480, (top_eqr/2))
                                 print("Positive transition btm_eqr is: ", btm_eqr)
                             else:
                                 print("No Positive transition eqr switch True  btm_eqr is: ", btm_eqr)
                         else:
                             check_btm = (1 - transition_rate480) * btm_eqr + transition_rate480 * scaled_value480
                             if check_eqr < check_btm:
                                 if check_eqr > top_eqr/2:
                                     btm_eqr = btm_eqr/2
                                 else:
                                     btm_eqr = btm_eqr
                             else: 
                                btm_eqr = check_btm
                             print("Negtive transition btm_eqr is: ", btm_eqr)
                         if -0.01 < btm_eqr < 0:
                             btm_eqr = 0
                         ttb10 = btm_eqr
                         self.portfolio['target_tier'] = 10
                         print("Under tier ",self.portfolio['target_tier'], "the new btm_eqr is: ", btm_eqr)
            if check_eqr >= 1:
                 btm_eqr = max(self.avg_check_eqr*.9,btm_eqr)
                 self.eqr_switch = True
            
            self.portfolio['top_eqr'] = top_eqr
            self.portfolio['btm_eqr'] = btm_eqr
            print(f"Position Target Tier is: {self.portfolio['target_tier']}")
            print("Top Equity Target is: ", top_eqr)
            print("Bottom Equity Target is: ", btm_eqr)
            print("Range Mode is: ", self.range_mode)
            
            rsi = self.data['RSI8'][self.current_step]
            obv = self.data['OBV'][self.current_step]
            short_ma = self.data['Short_MA'][self.current_step]
            long_ma = self.data['Long_MA'][self.current_step]
            
            if action == 0 or action == 1:
                if len(self.positions) >= 1:
                    action = 4
            
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
            total_actions = sell_count + buy_count
            if sell_count != 0:
              sc_per = sell_count/total_actions
            else:
              sc_per = 0
            if buy_count != 0:
              bc_per = buy_count/total_actions
            else:
              bc_per = 0
            if len(self.positions) <= 0:
                if abs(bc_per - sc_per) > .025:
                  if sc_per > bc_per:
                    if action != 6:
                      action = 0
                      print("Forced trade bias action in training...")
                  else:
                    if action != 6:
                      action = 1
                      print("Forced trade bias action in training...")

            if action == 5:
              tfmc = self.portfolio['total_free_margin']
              check_bet = (self.portfolio['total_balance']-tfmc)
              if start_port != 0 and check_bet != 0:
                check_eqr = start_port/check_bet
              else:
                check_eqr = 0
              if btm_eqr < check_eqr < top_eqr:
                action = 4
                action_factor = -1
            
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
              if eqrs == 1:
                if check_eqr < self.prev_eqr:
                  action = 5
                  eqrs = 0
                  save_eqrs(eqrs, eqrs_status_filename)
            
            if self.range_mode == True:
                if check_eqr >= top_eqr:
                    action = 5
                    print("Range Mode force closing at top eqr...")
            
            if hold_fct >= 1:
                action = 5
                action_factor = -1
            
            if action == 0:
              buy_count += 1
            if action == 1:
              sell_count += 1
              
            save_sell_count(sell_count, sc_filename)
            save_buy_count(buy_count, bc_filename)
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
            
            elif action == 4:  # Hold position (do nothing)
                for position in self.positions:
                    if position['type'] == 'buy':
                         position['current_profit'] = ((self.data['Close'][self.current_step] - position['open_price'])*10000) * (position['position_size'])
                    elif position['type'] == 'sell':
                         position['current_profit'] = ((position['open_price'] - self.data['Close'][self.current_step])*10000) * (position['position_size'])

            elif action == 5:  # Close all open buy and sell options
                positions_to_remove = []
                if 0 <= check_eqr < .10:
                    retain_amt = .90
                    vault_dis = .10
                elif 0.10 <= check_eqr < .20:
                    retain_amt = .80
                    vault_dis = .20
                elif 0.20 <= check_eqr < .30:
                    retain_amt = .70
                    vault_dis = .30
                elif 0.30 <= check_eqr < .40:
                    retain_amt = .60
                    vault_dis = .40
                elif 0.40 <= check_eqr < .50:
                    retain_amt = .50
                    vault_dis = .50
                elif 0.50 <= check_eqr < .60:
                    retain_amt = .40
                    vault_dis = .60
                elif 0.60 <= check_eqr < .70:
                    retain_amt = .30
                    vault_dis = .70
                elif 0.70 <= check_eqr < .80:
                    retain_amt = .20
                    vault_dis = .80
                elif check_eqr >= .80:
                    retain_amt = .10
                    vault_dis = .90  
                    
                for position in self.positions:
                    if position['type'] == 'buy':
                        buy_profit = position['current_profit']
                        buy_profit_amt = position['current_profit']
                        if buy_profit > 0:
                            buy_profit_amt = buy_profit * retain_amt
                            vault_amt = buy_profit * vault_dis
                            self.portfolio['vault'] += vault_amt
                        self.portfolio['total_balance'] += buy_profit_amt
                        self.portfolio['total_free_margin'] += buy_profit_amt
                        self.portfolio['total_free_margin'] += position['total_margin']
                        #self.positions.remove(position)
                        self.portfolio['total_buy_options'] -= 1
                    elif position['type'] == 'sell':
                        sell_profit = position['current_profit']
                        sell_profit_amt = position['current_profit']
                        if sell_profit > 0:
                            sell_profit_amt = sell_profit * retain_amt
                            vault_amt = sell_profit * vault_dis
                            self.portfolio['vault'] += vault_amt
                        self.portfolio['total_balance'] += sell_profit_amt
                        self.portfolio['total_free_margin'] += sell_profit_amt
                        self.portfolio['total_free_margin'] += position['total_margin']
                        #self.positions.remove(position)
                        self.portfolio['total_sell_options'] -= 1
                
                    positions_to_remove.append(position)
                
                for position in positions_to_remove:
                    self.positions.remove(position)    
            
            if self.positions:
                print("Position size:", self.positions[0]['position_size'])
            else:
                print("No open position yet.")

            self.portfolio['total_equity'] = self.portfolio['total_balance']
            for position in self.positions:
                pt = position['current_profit']
                self.portfolio['total_equity'] += pt
            print("-------------------------------------------------------------")
            #positioning life span 
            if len(self.positions) == 0:
              if action == 5:
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
                if holding_period != 0:
                    holding_period_factor = (holding_period / max_holding_period)# Adjust this as needed
                else:
                    holding_period_factor = 0
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
            if start_port != 0 and end_port != 0:
                reward_port = (end_port + start_port)/2
            else:
                reward_port = (end_port + start_port)
            self.portfolio['end_port'] = end_port
            if self.portfolio['reward_port'] != 0 and start_bet != 0:
                self.portfolio['reward_port'] = reward_port/start_bet
            else:
                self.portfolio['reward_port'] = reward_port
            rwp = 0
            if reward_port > 0:
              if agent_switch == 1:
                if len(self.positions)==0:
                  reward += 0
                  rwp += 0
                else:
                  if reward_port != 0 and start_bet != 0:
                      reward += reward_port/start_bet
                      rwp += reward_port/start_bet
                  else: 
                      reward += 0
                      rwp += 0
              else:
                if len(self.positions)==0:
                  reward2 += 0
                  rwp += 0
                else:
                  if reward_port != 0 and start_bet != 0:
                      reward2 += reward_port/start_bet
                      rwp += reward_port/start_bet
                  else: 
                      reward2 += 0
                      rwp += 0
            elif reward_port < 0:
              if agent_switch == 1:
                if len(self.positions)==0:
                   reward += 0
                   rwp += 0
                else:
                  if reward_port != 0 and start_bet != 0:
                      reward += reward_port/start_bet
                      rwp += reward_port/start_bet
                  else: 
                      reward += 0
                      rwp += 0
              else:
                if len(self.positions)==0:
                   reward2 += 0
                   rwp += 0
                else:
                  if reward_port != 0 and start_bet != 0:
                      reward2 += reward_port/start_bet
                      rwp += reward_port/start_bet
                  else: 
                      reward2 += 0
                      rwp += 0
            else:
              reward += 0
              reward2 += 0
            print(f"Total Equity Reward/Penalty: {rwp}")
            self.portfolio['rwp'] = rwp
            # Portfolio Rewards ----
            holdings = 0
            hdr_diff = (self.portfolio['total_balance']-10000)
            if hdr_diff != 0:
                hdr = hdr_diff / 10000
            else:
                hdr = 0
            if self.portfolio['total_balance'] < 10000:
                if agent_switch == 1:
                  if len(self.positions)==0:
                     holdings += 0
                     reward += 0
                  else:
                    holdings += hdr
                    reward += hdr
                else:
                  if len(self.positions)==0:
                     holdings += 0
                     reward2 += 0
                  else:
                    holdings += hdr
                    reward2 += hdr
            elif self.portfolio['total_balance'] >= 10000:
                if agent_switch == 1:
                  if len(self.positions)==0:
                     holdings += 0
                     reward += 0
                  else:
                    holdings += hdr
                    reward += hdr
                else:
                  if len(self.positions)==0:
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
            port_return_rate = portfolio_reward/(start_bet+ 1e-6)
            # Update the reward based on the change in total balance
            pr = 0
            if portfolio_reward != 0:
                port_rd = (port_return_rate / (self.portfolio['lifespan'] + 1e-6)) + (portfolio_reward * 0.0001)
            else:
                port_rd = 0
            if portfolio_reward > 0:
              if agent_switch == 1:
                if len(self.positions)==0:
                   pr += 0
                   reward += 0
                else:
                   pr += port_rd 
                   reward += port_rd 
              else:
                if len(self.positions)==0:
                   pr += 0
                   reward2 += 0
                else:
                  pr += port_rd 
                  reward2 += port_rd 
            elif portfolio_reward < 0:
              if agent_switch == 1:
                if len(self.positions)==0:
                   pr += 0
                   reward += 0
                else:
                  pr += port_rd
                  reward += port_rd
              else:
                if len(self.positions)==0:
                   pr += 0
                   reward2 += 0
                else:
                  pr += port_rd
                  reward2 += port_rd
            else:
              reward += 0
              reward2 += 0
            print("Closing Portfolio Balance Reward/Penalty:", pr)
            self.portfolio['pr'] = pr
            afrw = 0
            if action_factor == 1:
              if agent_switch == 1:
                if len(self.positions)==0:
                   reward += 0
                   afrw +=0
                else:
                  reward += 0.01
                  afrw += 0.01
              else:
                if len(self.positions)==0:
                   reward2 += 0
                   afrw += 0
                else:
                  reward2 += 0.01
                  afrw += 0.01
            else:
              if agent_switch == 1:
                if len(self.positions)==0:
                   reward += 0
                   afrw += 0
                else:
                  reward += -0.01
                  afrw += -0.01
              else:
                if len(self.positions)==0:
                   reward += 0
                   afrw += 0
                else:
                  afrw += -0.01
                  reward2 += -0.01
            print("Action Correction Reward/Penalty", afrw)
            total_rw = pr + holdings + rwp + afrw
            print("--------------------------------------------------------")
            # Penalty if balance is below minimum threshold
            if action == 0 or action == 1:
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
            bet = end_bet
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
            trade_bias = abs(bc_per - sc_per)
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
            if action == 0 or action == 1:
              print(f"Risk Ratio is: {risk_ratio}")
            else:
              print(f"Risk Ratio is: {start_risk_ratio}")
            print(f"Risk Ratio Factor: {risk_penalty}")
            print(f"Overall Equity Risk Ratio Factor: {pos_rw}")
            print("--------------------------------------------------------")
            if hf != 0:
                print("Total Step Holding/Period Factor:", hf)
                if len(self.positions)==0:
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
                  if len(self.positions)==0:
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
                  if len(self.positions)==0:
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
                      
            
            hold_penalty = 0
            if action == 4 and len(self.positions) > 0:
                self.hold_count += 1

                # Scale based on portfolio baseline (like other rewards)
                scaled_penalty = (0.001 * self.hold_count) * (self.portfolio['total_balance'] / 10000)

                hold_penalty = -scaled_penalty

                if agent_switch == 1:
                    reward += hold_penalty
                else:
                    reward2 += hold_penalty

                total_rw += hold_penalty  # Track in total_rw like other components
                print("Hold penalty with position open is: ", hold_penalty)
            else:
                self.hold_count = 0
            
            fault = False
            if self.portfolio['total_equity'] > (10000 * 0.90) or self.portfolio['total_balance'] > (10000 * 0.90):
                scaled_step = 0.0001 * self.current_step 
                if agent_switch == 1:
                  self.reset1 += 0
                  reward += scaled_step
                  total_rw += scaled_step
                else:
                  self.reset2 += 0
                  reward2 += scaled_step
                  total_rw += scaled_step
                print(f"scaled step reward {scaled_step}")
                  
            if self.portfolio['total_equity'] < (10000 * 0.90) or self.portfolio['total_balance'] < (10000 * 0.90):
                fault = True
                scaled_step = ((0.0001 * self.current_step)*self.current_step)*-1
                if agent_switch == 1:
                  self.reset1 += 1
                  reward += scaled_step
                  total_rw += scaled_step
                else:
                  self.reset2 += 1
                  reward2 += scaled_step
                  total_rw += scaled_step
                print(f"scaled step penalty 1 {scaled_step}")
            
            if fault == False:         
              if not self.positions and self.portfolio['total_balance'] < (10000 * 1):
                fault = True
                scaled_step = ((0.0001 * self.current_step)*self.current_step)*-1
                if agent_switch == 1:
                  self.reset1 += 1
                  reward += scaled_step
                  total_rw += scaled_step
                else:
                  self.reset2 += 1
                  reward2 += scaled_step
                  total_rw += scaled_step
                print(f"scaled step penalty 2 {scaled_step}")

            #if fault == False:
            #  if agent_switch == 1:
            #    if len(self.positions)==0 and action == 4:
            #      reward += 0
            #    else:
            #      reward += 1
            #  else:
            #    if len(self.positions)==0 and action == 4:
            #      reward2 += 0
            #    else:
            #      reward2 += 1
            
            if self.lifespan == 0 :
                  pacer_eqr = 0
            else:  
                  pacer_eqr = self.avg_check_eqr/self.lifespan
            
            if pacer_eqr < .02/96 and self.lifespan >= 1:
                print(f"Pace is at:  {pacer_eqr}")
                if agent_switch == 1:
                    reward += pacer_eqr * self.lifespan
                    total_rw += pacer_eqr * self.lifespan
                else:
                    reward2 += pacer_eqr * self.lifespan
                    total_rw += pacer_eqr * self.lifespan
            elif pacer_eqr > .02/96 and self.lifespan >= 1:
                print(f"Pace is at:  {pacer_eqr}")
                if agent_switch == 1:
                    reward += pacer_eqr * self.lifespan
                    total_rw += pacer_eqr * self.lifespan
                else:
                    reward2 += pacer_eqr * self.lifespan
                    total_rw += pacer_eqr * self.lifespan
            
            flip_bonus = 0
            if action in [0, 1]:
                if self.last_open_action is not None:
                    if action != self.last_open_action:
                        if agent_switch == 1:
                            if total_rw > 0:
                                flip_bonus += abs(total_rw) * 0.50
                            else:
                                flip_bonus -= abs(total_rw) * 0.25
                            reward += flip_bonus
                            total_rw += flip_bonus
                        else:
                            if total_rw > 0:
                                flip_bonus += abs(total_rw) * 0.50
                            else:
                                flip_bonus -= abs(total_rw) * 0.25
                            reward2 += flip_bonus
                            total_rw += flip_bonus
                    if action == self.last_open_action:
                        if agent_switch == 1:
                            flip_bonus -= abs(total_rw) * 0.125
                            reward += flip_bonus
                            total_rw += flip_bonus
                        else:
                            flip_bonus -= abs(total_rw) * 0.125
                            reward2 += flip_bonus
                            total_rw += flip_bonus
                self.last_open_action = action
                
            q_file = "q_values_select_log.csv"
            if os.path.getsize(q_file) > 0:
                df = pd.read_csv(q_file, header=None)
                last_q_row = ast.literal_eval(df.iloc[-1, 0])
                q0, q1, q2, q3 = last_q_row  # or however many actions you have
            else: 
                q0, q1, q2, q3 = 0,0,0,0
            threshold = 0.1
            penalty_boost = 0.01  # how much to boost epsilon per low-spread event
            if agent_switch == 1:
                if dqn_agent.epsilon > dqn_agent.epsilon_min:
                    penalty_boost = dqn_agent.epsilon * (1 - dqn_agent.epsilon_decay)
            else:
                if dqn_agent.epsilon2 > dqn_agent.epsilon_min:
                    penalty_boost = dqn_agent.epsilon2 * (1 - dqn_agent.epsilon_decay)
            decayed_this_step = False
            #boosted_this_step = False
            if agent_switch == 1:
                if len(self.positions) == 0:
                   # Entry situation (0, 1, hold w/ no position)
                   spread = max(q0, q1) - min(q0, q1)
                   if spread < threshold:
                       if self.current_step - self.last_boost_step >= 10:
                           if total_rw > 0:
                               reward -= abs(total_rw) * 0.5
                               total_rw -= abs(total_rw) * 0.5
                           dqn_agent.epsilon = min(dqn_agent.epsilon + penalty_boost, .99)
                           #boosted_this_step = True
                           print(f"[Exploration Boost] Q-spread too narrow: {spread:.3f}, epsilon now {dqn_agent.epsilon:.3f}")
                       else:
                           print(f"[Boost Skipped] Too soon since last boost at step {self.last_boost_step}")
                   else:
                       if total_rw > 0:
                           dqn_agent.epi_decay()
                           decayed_this_step = True
                           print(f"[Exploration Decay] Q-spread Wide: {spread:.3f}, epsilon now {dqn_agent.epsilon:.3f}")
                       else:
                           dqn_agent.epsilon = min(dqn_agent.epsilon + penalty_boost, .99)
                           print(f"[Exploration Boost] Q-spread too narrow: {spread:.3f}, epsilon now {dqn_agent.epsilon:.3f}")
                else:
                   # Position is open ??? close vs hold
                   spread = max(q2, q3) - min(q2, q3)
                   if spread < threshold:
                       if self.current_step - self.last_boost_step >= 10:
                           if total_rw > 0:
                               reward -= abs(total_rw) * 0.5
                               total_rw -= abs(total_rw) * 0.5
                           dqn_agent.epsilon = min(dqn_agent.epsilon + penalty_boost, .99)
                           #boosted_this_step = True
                           print(f"[Exploration Boost] Q-spread too narrow: {spread:.3f}, epsilon now {dqn_agent.epsilon:.3f}")
                       else:
                           print(f"[Boost Skipped] Too soon since last boost at step {self.last_boost_step}")
                   else:
                       if total_rw > 0:
                           dqn_agent.epi_decay()
                           decayed_this_step = True
                           print(f"[Exploration Decay] Q-spread Wide: {spread:.3f}, epsilon now {dqn_agent.epsilon:.3f}")
                       else:
                           dqn_agent.epsilon = min(dqn_agent.epsilon + penalty_boost, .99)
                           print(f"[Exploration Boost] Q-spread too narrow: {spread:.3f}, epsilon now {dqn_agent.epsilon:.3f}")
            else:
                if len(self.positions) == 0:
                   # Entry situation (0, 1, hold w/ no position)
                   spread = max(q0, q1) - min(q0, q1)
                   if spread < threshold:
                       if self.current_step - self.last_boost_step >= 10:
                           if total_rw > 0:
                               reward2 -= abs(total_rw) * 0.5
                               total_rw -= abs(total_rw) * 0.5
                           dqn_agent.epsilon2 = min(dqn_agent.epsilon2 + penalty_boost, .99)
                           #boosted_this_step = True
                           print(f"[Exploration Boost] Q-spread too narrow: {spread:.3f}, epsilon now {dqn_agent.epsilon2:.3f}")
                       else:
                           print(f"[Boost Skipped] Too soon since last boost at step {self.last_boost_step}")
                   else:
                       if total_rw > 0:
                           dqn_agent.epi_decay()
                           decayed_this_step = True
                           print(f"[Exploration Decay] Q-spread Wide: {spread:.3f}, epsilon now {dqn_agent.epsilon:.3f}")
                       else:
                           dqn_agent.epsilon = min(dqn_agent.epsilon + penalty_boost, .99)
                           print(f"[Exploration Boost] Q-spread too narrow: {spread:.3f}, epsilon now {dqn_agent.epsilon:.3f}")
                else:
                   # Position is open ??? close vs hold
                   spread = max(q2, q3) - min(q2, q3)
                   if spread < threshold:
                       if self.current_step - self.last_boost_step >= 10:
                           if total_rw > 0:
                               reward2 -= abs(total_rw) * 0.5
                               total_rw -= abs(total_rw) * 0.5
                           dqn_agent.epsilon2 = min(dqn_agent.epsilon2 + penalty_boost, .99)
                           #boosted_this_step = True
                           print(f"[Exploration Boost] Q-spread too narrow: {spread:.3f}, epsilon now {dqn_agent.epsilon2:.3f}")
                       else:
                           print(f"[Boost Skipped] Too soon since last boost at step {self.last_boost_step}")
                   else:
                       if total_rw > 0:
                           dqn_agent.epi_decay()
                           decayed_this_step = True
                           print(f"[Exploration Decay] Q-spread Wide: {spread:.3f}, epsilon now {dqn_agent.epsilon:.3f}")
                       else:
                           dqn_agent.epsilon = min(dqn_agent.epsilon + penalty_boost, .99)
                           print(f"[Exploration Boost] Q-spread too narrow: {spread:.3f}, epsilon now {dqn_agent.epsilon:.3f}")
            
            #if not boosted_this_step:
            #    if agent_switch == 1:
            #        dqn_agent.epsilon = min(dqn_agent.epsilon + penalty_boost, .99)
            #        print(f"[Exploration Boost] for Step...")
            #    else:
            #        dqn_agent.epsilon2 = min(dqn_agent.epsilon2 + penalty_boost, .99)
            #        print(f"[Exploration Boost] for Step...")
            if not decayed_this_step:
                dqn_agent.epi_decay()
                print(f"[Exploration Decay] for Step...")
                
                        
            if len(self.positions) == 0 and action == 5:
                if self.portfolio['vault'] > 0:
                    vault_rw = self.portfolio['vault']/self.portfolio['total_balance']
                    if agent_switch == 1:
                        reward += vault_rw 
                        total_rw += vault_rw
                    else:
                        reward2 += vault_rw 
                        total_rw += vault_rw
                        
            self.portfolio['total_rw'] = total_rw
            self.portfolio['reward'] = reward
            self.portfolio['reward2'] = reward2
            
            with open("reward_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([agent_switch, reward, reward2, total_rw])
                
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
            print("Total Vault:", self.portfolio['vault'])
            print("Total Free Margin:", self.portfolio['total_free_margin'])
            print("Total Equity:", self.portfolio['total_equity'])
            print("Total Buy Options:", self.portfolio['total_buy_options'])
            print("Total Profit from Open Buy Options:", total_buy_profit)
            print("Total Sell Options:", self.portfolio['total_sell_options'])
            print("Total Profit from Open Sell Options:", total_sell_profit)
            print("Total Profit from All Open Options:", total_open_positions_profit)
            
            if action == 5 and check_eqr <= 0:
                self.portfolio['recent_losses'] += 1
            if action == 5 and check_eqr > 0:
                if self.portfolio['recent_losses'] >= 1:
                    self.portfolio['recent_losses'] = 0
            
            print("consecutive losses are: ", self.portfolio['recent_losses'])
            
            total_bal = self.portfolio['total_balance']
            # Update current step
            self.current_step += 1
            self.prev_eqr = eqr 
            # Determine if the episode is done (you need to define your own termination conditions)
            
            done = False

            # Episode ends when reaching data boundary
            if self.current_step >= len(self.data) - 1:
                done = True

            # Episode ends if account balance drops too low
            if self.portfolio['total_balance'] < 10000 * 0.9 or self.portfolio['total_equity'] < (10000 * 0.90):
                done = True

            return next_state, reward, reward2, total_bal, original_action, action_factor, self.reset1, self.reset2,trade_bias, action, total_rw, done
    
    # DQN Agent
    class DQNAgent:
        def __init__(self, state_size, action_size, env, model_filename=None, model_filename2=None,memory_filename=None, memory_filename2=None):
            self.state_size = state_size
            self.action_size = action_size
            self.memory = deque(maxlen=max_mem)
            self.memory2 = deque(maxlen=max_mem)
            self.gamma = 0.95  # Discount factor
            self.epsilon = epi  # Exploration rate
            self.epsilon2 = epi2  # Exploration rate
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.999
            self.learning_rate = 0.001
            self.model = self._build_model()
            self.model2 = self._build_model()
            self.env = env  # Store the environment instance
            self.model_filename = model_filename
            self.model_filename2 = model_filename2
            self.memory_filename = memory_filename
            self.memory_filename2 = memory_filename2
            self.failed_batch_count = 0
            self.valid_batch_threshold = 1
            self.last_opening_action = 0
            self.last_random_action = 1
            
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
                    print("Could not load model2 weights. Starting with random weights.")
            # Load memory if it exists
            if memory_filename:
                try:
                    self.load_memory(memory_filename)
                    print("Loaded memory from", memory_filename)
                except:
                    print("Could not load memory. Starting with empty memory.")
                        # Load memory if it exists
            if memory_filename2:
                try:
                    self.load_memory2(memory_filename2)
                    print("Loaded memory2 from", memory_filename2)
                except:
                    print("Could not load memory2. Starting with empty memory2.")  
                    
        def _build_model(self):
           input_layer = Input(shape=(self.state_size,))
           shared_layer1 = Dense(128, activation='relu')(input_layer)
           shared_layer2 = Dense(256, activation='relu')(shared_layer1)
           reshaped_state = Reshape((1, 256))(shared_layer2)
           # Transformer Layer
           num_transformer_layers = 1
           transformer_output = reshaped_state
           for _ in range(num_transformer_layers):
             transformer_output = MultiHeadAttention(num_heads=32, key_dim=self.state_size // 64)(transformer_output, transformer_output)
             transformer_output = LayerNormalization()(transformer_output)
             transformer_output = Dropout(0.1)(transformer_output)
           transformer_output = Flatten()(transformer_output)
           
           num_transformer_layers = 2
           transformer_output2 = reshaped_state
           for _ in range(num_transformer_layers):
             transformer_output2 = MultiHeadAttention(num_heads=32, key_dim=self.state_size // 64)(transformer_output2, transformer_output2)
             transformer_output2 = LayerNormalization()(transformer_output2)
             transformer_output2 = Dropout(0.1)(transformer_output2)
           transformer_output2 = Flatten()(transformer_output2)
             
           num_transformer_layers = 3
           transformer_output3 = reshaped_state
           for _ in range(num_transformer_layers):
             transformer_output3 = MultiHeadAttention(num_heads=32, key_dim=self.state_size // 64)(transformer_output3, transformer_output3)
             transformer_output3 = LayerNormalization()(transformer_output3)
             transformer_output3 = Dropout(0.1)(transformer_output3)
           transformer_output3 = Flatten()(transformer_output3)
             
           num_transformer_layers = 1
           transformer_output4 = reshaped_state
           for _ in range(num_transformer_layers):
             transformer_output4 = MultiHeadAttention(num_heads=32, key_dim=self.state_size // 64)(transformer_output4, transformer_output4)
             transformer_output4 = LayerNormalization()(transformer_output4)
             transformer_output4 = Dropout(0.1)(transformer_output4)
           transformer_output4 = Flatten()(transformer_output4)
             
           num_transformer_layers = 2
           transformer_output5 = reshaped_state
           for _ in range(num_transformer_layers):
             transformer_output5 = MultiHeadAttention(num_heads=32, key_dim=self.state_size // 64)(transformer_output5, transformer_output5)
             transformer_output5 = LayerNormalization()(transformer_output5)
             transformer_output5 = Dropout(0.1)(transformer_output5)
           transformer_output5 = Flatten()(transformer_output5)
             
           num_transformer_layers = 3
           transformer_output6 = reshaped_state
           for _ in range(num_transformer_layers):
             transformer_output6 = MultiHeadAttention(num_heads=32, key_dim=self.state_size // 64)(transformer_output6, transformer_output6)
             transformer_output6 = LayerNormalization()(transformer_output6)
             transformer_output6 = Dropout(0.1)(transformer_output6)
           transformer_output6 = Flatten()(transformer_output6)
           # Value Streams
           value_stream1 = Dense(256, activation='relu')(transformer_output)
           value_stream1 = LayerNormalization()(value_stream1)
           value1 = Dense(1, activation='linear')(value_stream1)

           value_stream2 = Dense(256, activation='relu')(transformer_output2)
           value_stream2 = LayerNormalization()(value_stream2)
           value2 = Dense(1, activation='linear')(value_stream2)
           
           value_stream3 = Dense(256, activation='relu')(transformer_output3)
           value_stream3 = LayerNormalization()(value_stream3)
           value3 = Dense(1, activation='linear')(value_stream3)
           
           value_stream4 = Dense(256, activation='relu')(transformer_output4)
           value_stream4 = LayerNormalization()(value_stream4)
           value4 = Dense(1, activation='linear')(value_stream4)
           
           value_stream5 = Dense(256, activation='relu')(transformer_output5)
           value_stream5 = LayerNormalization()(value_stream5)
           value5 = Dense(1, activation='linear')(value_stream5)
           
           value_stream6 = Dense(256, activation='relu')(transformer_output6)
           value_stream6 = LayerNormalization()(value_stream6)
           value6 = Dense(1, activation='linear')(value_stream6)

           # Advantage Streams
           advantage_stream1 = Dense(256, activation='relu')(transformer_output)
           advantage_stream1 = LayerNormalization()(advantage_stream1)
           advantage1 = Dense(self.action_size, activation='linear')(advantage_stream1)

           advantage_stream2 = Dense(256, activation='relu')(transformer_output2)
           advantage_stream2 = LayerNormalization()(advantage_stream2)
           advantage2 = Dense(self.action_size, activation='linear')(advantage_stream2)
           
           advantage_stream3 = Dense(256, activation='relu')(transformer_output3)
           advantage_stream3 = LayerNormalization()(advantage_stream3)
           advantage3 = Dense(self.action_size, activation='linear')(advantage_stream3)
           
           advantage_stream4 = Dense(256, activation='relu')(transformer_output4)
           advantage_stream4 = LayerNormalization()(advantage_stream4)
           advantage4 = Dense(self.action_size, activation='linear')(advantage_stream4)
           
           advantage_stream5 = Dense(256, activation='relu')(transformer_output5)
           advantage_stream5 = LayerNormalization()(advantage_stream5)
           advantage5 = Dense(self.action_size, activation='linear')(advantage_stream5)
           
           advantage_stream6 = Dense(256, activation='relu')(transformer_output6)
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
         
        def remember(self, state, original_action, action, reward, next_state, total_rw, done):
            if agent_switch == 1:
              timestamp = int(datetime.utcnow().strftime('%Y%m%d%H%M%S'))  # e.g., 20250501124556
              self.memory.append((state, original_action, action, reward, next_state, total_rw, timestamp, done))
            else:
              timestamp = int(datetime.utcnow().strftime('%Y%m%d%H%M%S'))  # e.g., 20250501124556
              self.memory2.append((state, original_action, action, reward, next_state, total_rw, timestamp, done))
        
        def save_memory(self, filename, action):
          print("The len of memory1 is ", len(self.memory))
          if action == 5:
            if len(self.memory) >= mem_check:
              timestamp = int(time.time())  # Get current timestamp
              np.save(f"mem_bank/memory_{timestamp}.npy", np.array(self.memory, dtype=object))
              os.remove(filename)  # Delete the memory file after saving
              self.memory = deque(maxlen=max_mem)
            else:
              np.save(filename, np.array(self.memory, dtype=object))
                
        def save_memory2(self, filename, action):
          print("The len of memory2 is ", len(self.memory2))
          if action == 5:
            if len(self.memory2) >= mem_check:
              timestamp = int(time.time())  # Get current timestamp
              np.save(f"mem_bank/memory2_{timestamp}.npy", np.array(self.memory2, dtype=object))
              os.remove(filename)  # Delete the memory file after saving
              self.memory2 = deque(maxlen=max_mem)
            else:
              np.save(filename, np.array(self.memory2, dtype=object))

        def load_memory(self, filename):
            self.memory = deque(np.load(filename, allow_pickle=True), maxlen=max_mem)
        
        def load_memory2(self, filename):
            self.memory2 = deque(np.load(filename, allow_pickle=True), maxlen=max_mem)
            
            
        def act(self, state):
            action_space_mapping = [0, 1, 4, 5]
            q_values = self.model.predict(state)
            q_values2 = self.model2.predict(state)
            
            q_val_list.append(q_values.tolist())  # convert to plain Python list before appending
            with open("q_values_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(q_values.tolist()) 
                    
            q_val_list2.append(q_values2.tolist())  # convert to plain Python list before appending
            with open("q_values2_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(q_values2.tolist()) 
                    
            conf1 = np.max(q_values) - np.mean(q_values)
            conf2 = np.max(q_values2) - np.mean(q_values2)
            if conf1 > conf2:
                q_values = q_values
                agent_switch = 1
            else:
                q_values = q_values2
                agent_switch = 2
            if abs(conf1 - conf2) < 0.05:
                q_values = (q_values + q_values2) / 2
                if agent_switch == 1:
                  agent_switch = 2
                else:
                  agent_switch = 1
            
            q_val_list_select.append(q_values.tolist())  # convert to plain Python list before appending
            with open("q_values_select_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(q_values.tolist())
                
            if agent_switch == 1:
              # Exploration: choose a random action if epsilon-greedy exploration
              if np.random.rand() <= self.epsilon:
                print("random action...")
                # Get a list of suitable actions based on open positions
                suitable_actions = []
            
                if not self.env.positions:# and trend_direction == 0:  # If there are no open positions
                    suitable_actions.extend([0,1])
                else:
                    if any(position['type'] == 'buy' for position in self.env.positions):# and trend_direction == 0:
                        suitable_actions.extend([4, 5])  # Add actions 2 and 5 if there are open buy positions
                    if any(position['type'] == 'sell' for position in self.env.positions):# and trend_direction == 0:
                        suitable_actions.extend([4, 5])  # Add actions 3 and 5 if there are open sell positions             
                    if any(position['type'] == 'buy' for position in self.env.positions) and any(position['type'] == 'sell' for position in self.env.positions):# and trend_direction == 0:
                        suitable_actions.extend([4, 5])
                suitable_actions = [action for action in suitable_actions if action in action_space_mapping]
                
                # Map suitable actions to their corresponding model indices
                model_indices = [action_space_mapping.index(action) for action in suitable_actions]
        
                # Filter Q-values to keep only those corresponding to suitable actions
                filtered_q_values = [q_values[0][index] for index in model_indices]
                # **Convert Q-values to probabilities using Softmax**
                #action_probabilities = tf.nn.softmax(filtered_q_values).numpy()
                # --- Convert Q-values to probabilities using Softmax (scipy version) ---
                softmax_input = np.array(filtered_q_values)
                softmax_probs = scipy.special.softmax(softmax_input)
                #print("filtered q-values are: ", filtered_q_values)
                #print("action probs are: ", action_probabilities)
                # **Get the best action and its probability**
                # --- Sample an action stochastically based on softmax probabilities ---
                sampled_index = np.random.choice(len(model_indices), p=softmax_probs)
                sampled_action = action_space_mapping[model_indices[sampled_index]]
                sampled_prob = softmax_probs[sampled_index]
                return np.random.choice(suitable_actions), sampled_prob
            else:
              # Exploration: choose a random action if epsilon-greedy exploration
              if np.random.rand() <= self.epsilon2:
                print("random action...")
                # Get a list of suitable actions based on open positions
                suitable_actions = []
            
                if not self.env.positions:# and trend_direction == 0:  # If there are no open positions
                    suitable_actions.extend([0,1])
                else:
                    if any(position['type'] == 'buy' for position in self.env.positions):# and trend_direction == 0:
                        suitable_actions.extend([4, 5])  # Add actions 2 and 5 if there are open buy positions
                    if any(position['type'] == 'sell' for position in self.env.positions):# and trend_direction == 0:
                        suitable_actions.extend([4, 5])  # Add actions 3 and 5 if there are open sell positions             
                    if any(position['type'] == 'buy' for position in self.env.positions) and any(position['type'] == 'sell' for position in self.env.positions):# and trend_direction == 0:
                        suitable_actions.extend([4, 5])
                suitable_actions = [action for action in suitable_actions if action in action_space_mapping]
                # Map suitable actions to their corresponding model indices
                model_indices = [action_space_mapping.index(action) for action in suitable_actions]
        
                # Filter Q-values to keep only those corresponding to suitable actions
                filtered_q_values = [q_values[0][index] for index in model_indices]
                # **Convert Q-values to probabilities using Softmax**
                #action_probabilities = tf.nn.softmax(filtered_q_values).numpy()
                # --- Convert Q-values to probabilities using Softmax (scipy version) ---
                softmax_input = np.array(filtered_q_values)
                softmax_probs = scipy.special.softmax(softmax_input)
                #print("filtered q-values are: ", filtered_q_values)
                #print("action probs are: ", action_probabilities)
                # **Get the best action and its probability**
                # --- Sample an action stochastically based on softmax probabilities ---
                sampled_index = np.random.choice(len(model_indices), p=softmax_probs)
                sampled_action = action_space_mapping[model_indices[sampled_index]]
                sampled_prob = softmax_probs[sampled_index]
                return np.random.choice(suitable_actions), sampled_prob
            
            print("model action...")
            # Get suitable actions based on open positions
            suitable_actions = []
            if not self.env.positions:# and trend_direction == 0:  # If there are no open positions
                suitable_actions.extend([0,1])  # Choose from actions 0, 1, or 4
            else:
                if any(position['type'] == 'buy' for position in self.env.positions):# and trend_direction == 0:
                    suitable_actions.extend([4, 5])  # Add actions 2 and 5 if there are open buy positions
                if any(position['type'] == 'sell' for position in self.env.positions):# and trend_direction == 0:
                    suitable_actions.extend([4, 5])  # Add actions 3 and 5 if there are open sell positions
                if any(position['type'] == 'buy' for position in self.env.positions) and any(position['type'] == 'sell' for position in self.env.positions):# and trend_direction == 0:
                    suitable_actions.extend([4, 5])
        
            # Filter suitable actions to ensure they are within the mapped space
            suitable_actions = [action for action in suitable_actions if action in action_space_mapping]

            # Map suitable actions to their corresponding model indices
            model_indices = [action_space_mapping.index(action) for action in suitable_actions]
    
            # Filter Q-values to keep only those corresponding to suitable actions
            filtered_q_values = [q_values[0][index] for index in model_indices]
            # **Convert Q-values to probabilities using Softmax**
            #action_probabilities = tf.nn.softmax(filtered_q_values).numpy()
            # --- Convert Q-values to probabilities using Softmax (scipy version) ---
            softmax_input = np.array(filtered_q_values)
            softmax_probs = scipy.special.softmax(softmax_input)
            #print("filtered q-values are: ", filtered_q_values)
            #print("action probs are: ", action_probabilities)
            # **Get the best action and its probability**
            # --- Sample an action stochastically based on softmax probabilities ---
            sampled_index = np.random.choice(len(model_indices), p=softmax_probs)
            sampled_action = action_space_mapping[model_indices[sampled_index]]
            sampled_prob = softmax_probs[sampled_index]
            
            # --- Log both best (argmax) and sampled actions ---
            best_action_index = np.argmax(filtered_q_values)
            best_action = action_space_mapping[model_indices[best_action_index]]
            best_action_probability = softmax_probs[best_action_index]

            print("Filtered Q-Values are: ", filtered_q_values)
            print(f"[Stochastic] Sampled: {sampled_action} (p={sampled_prob:.2f}), Best: {best_action} (p={best_action_probability:.2f})")

            # **Filtering based on probability threshold ONLY for new trades**
            probability_threshold = 0.7
            print(f"Action {sampled_action} is @ probability ({sampled_prob:.2f}).")
            if not self.env.positions and sampled_action in [0, 1]:  # Only apply if no open positions and action is 0 or 1
                if sampled_prob < probability_threshold:
                    print(f"Action {sampled_action} rejected due to low probability ({sampled_prob:.2f}).")
                    action = 0 if self.last_random_action == 1 else 1
                    self.last_random_action = action
                    return action, sampled_prob  # Default to action 4 if confidence is too low

            # **Return the selected action**
            return sampled_action, sampled_prob
        
        def epi_decay(self):
          if agent_switch == 1:
              if self.epsilon > self.epsilon_min:
                  self.epsilon *= self.epsilon_decay
              return self.epsilon
          else:
            if self.epsilon2 > self.epsilon_min:
                self.epsilon2 *= self.epsilon_decay
            return self.epsilon2
          
        def replay(self, batch_size, batch_size1, batch_size2, batch_size3, batch_size4, batch_size5, batch_size6, agent_switch):
            import random
            random_number = random.randint(1,7)
            bz = batch_size
            if random_number == 1:
              bz = batch_size 
            elif random_number == 2:
              bz = batch_size1
            elif random_number == 3:
              bz = batch_size2
            elif random_number == 4:
              bz = batch_size3
            elif random_number == 5:
              bz = batch_size4
            elif random_number == 6:
              bz = batch_size5
            elif random_number == 7:
              bz = batch_size6
              
            # Define action space mapping
            action_space_mapping = [0, 1, 4, 5]  # Valid actions
            inverse_mapping = {action: idx for idx, action in enumerate(action_space_mapping)}  # Reverse mapping

            if agent_switch == 1:
              if len(self.memory) < bz:
                print("not training the model on sequential memory for ... agent-",agent_switch)
                return
              
              print("training the model on sequential memory ", bz ,"... agent-",agent_switch)
              #batch = random.sample(self.memory, bz)
              # Convert the deque to a list to enable slicing
              memory_list = list(self.memory)
              # === Determine batch selection strategy ===
              use_valid_batch = random.random() < 0.25  # 25% chance to try the valid open???close batch

              if use_valid_batch:
                  # === Your original logic to find a valid open???close batch ===
                  max_retries = 5
                  attempts = 0
                  batch = None

                  preferred_open_action = 1 if self.last_opening_action == 0 else 0

                  while batch is None and attempts < max_retries:
                      attempts += 1
                      start_index = None

                      for i in range(len(memory_list)):
                          action = memory_list[i][2]
                          if action == preferred_open_action:
                              start_index = i
                              self.last_opening_action = preferred_open_action
                              break

                      if start_index is not None:
                          end_index = min(start_index + bz, len(memory_list))
                          batch = memory_list[start_index:end_index]
                          actions = [x[2] for x in batch]

                          if any(a in [5] for a in actions):
                              rewards = [x[3] for x in batch]

                              hold_start = None
                              for i in range(len(actions) - 1, -1, -1):
                                  if actions[i] != 4:
                                      break
                                  hold_start = i

                              if hold_start is not None:
                                  trailing_hold_rewards = rewards[hold_start:]
                                  reward_sum = sum(trailing_hold_rewards)

                                  if reward_sum <= 0 and len(trailing_hold_rewards) >= 3:
                                      print(f"Attempt {attempts}: Trailing hold with no reward (len={len(trailing_hold_rewards)}, reward={reward_sum}) - retrying...")
                                      batch = None
                                      continue
                                  else:
                                      print(f"Attempt {attempts}: Valid open???close batch found.")
                                      self.failed_batch_count = 0
                                      break
                              else:
                                  print(f"Attempt {attempts}: Valid open???close batch found (no hold detected).")
                                  self.failed_batch_count = 0
                                  break
                          else:
                              print(f"Attempt {attempts}: No closing action found - retrying...")
                              batch = None
                      else:
                          print(f"Attempt {attempts}: No valid opening action ({preferred_open_action}) found - retrying...")
                          batch = None

                  # Fallback if validation fails
                  if batch is None:
                      print("Fallback to random batch due to failed valid batch attempts.")
                      batch = stratified_random_batch(memory_list, bz)


              else:
                  # === 75% of the time: train on a random batch immediately ===
                  print("Random Batch Training (75% chance)")
                  batch = stratified_random_batch(memory_list, bz)
              
              skip_status = False
              if batch:
                batch_hash = hash_batch(batch)
                if batch_hash in trained_batches:
                  print("[SKIP] Batch already trained on - retrying...")
                  batch = None  # force retry/skip
                  skip_status = True
                else:
                  # Similarity check
                  new_vec = compute_batch_vector(batch).reshape(1, -1)
                  similar = False
                  for past_vec in trained_batches.values():
                      if isinstance(past_vec, str):  # Fix for string case
                        past_vec = np.fromstring(past_vec.strip('[]'), sep=',')
                      past_vec = np.asarray(past_vec).flatten()
                      if len(past_vec) < new_vec.shape[1]:
                        past_vec = np.pad(past_vec, (0, new_vec.shape[1] - len(past_vec)))
                      else:
                        past_vec = past_vec[:new_vec.shape[1]]
                      sim = hybrid_similarity(new_vec.flatten(), past_vec.flatten())
                      if sim > similarity_threshold:
                          print(f"[SKIP] Similar batch found (similarity={sim:.3f}) - skipping.")
                          similar = True
                          batch = None
                          skip_status = True
        
                  if not similar and batch is not None:
                      print("[ACCEPTED] Batch is unique and dissimilar - training.")
                      trained_batches[batch_hash] = new_vec.flatten()
                      with open(trained_batches_file, "wb") as f:
                          pickle.dump(trained_batches, f)
                  else:
                    print("[SKIP] No valid batch selected - skipping training.")
                    skip_status = True
              
              if skip_status == False:
                  # Create a sequential batch of elements from the memory
                  states, original_actions, actions, rewards, next_states, total_rw, timestamp, dones = zip(*batch)
                  states = np.vstack(states)
                  next_states = np.vstack(next_states)
                  rewards = np.array(total_rw)
                  reward_mean = np.mean(rewards)
                  reward_std = np.std(rewards) + 1e-6  # avoid div-by-zero
                  rewards = (rewards - reward_mean) / reward_std
                  dones = np.array(dones).astype(int)  # True = 1, False = 0
                  
                  # Step 1: Predict Q-values from online model (to get best action indices)
                  next_q_values_online = self.model.predict(next_states, verbose=0)
                  next_actions = np.argmax(next_q_values_online, axis=1)
                  
                  # Step 2: Predict Q-values from target model (to evaluate selected actions)
                  next_q_values_target = self.model2.predict(next_states, verbose=0)
                  next_q_selected = next_q_values_target[np.arange(len(next_states)), next_actions]
                  
                  # Step 3: Compute targets using Double DQN equation
                  targets = rewards + self.gamma * next_q_selected * (1 - dones)
                  
                  # Step 4: Predict current Q-values and apply update to taken actions
                  mapped_actions = [inverse_mapping[action] for action in actions]
                  target_values = self.model.predict(states, verbose=0)
                  target_values[np.arange(len(target_values)), mapped_actions] = targets
                  
                  # Step 5: Train the model
                  self.model.fit(states, target_values, epochs=1, verbose=0)
              
              mem_bank_dir = "mem_bank"
              mem_files = os.listdir(mem_bank_dir)
    
              if len(mem_files) <= 1:
              #  if self.epsilon > self.epsilon_min:
              #     self.epsilon *= self.epsilon_decay
                print("not training the model on memory bank for ... agent-",agent_switch)
                return   # No need to train if only one or no memory files
    
              selected_file = random.choice(mem_files)
              file_path = os.path.join(mem_bank_dir, selected_file)
    
              # Load the memory from the selected file
              memory = deque(np.load(file_path, allow_pickle=True), maxlen=max_mem)
              
              print("training the model on memory bank ",bz,"... agent-", agent_switch)
              # Train the model using the loaded memory
              #batch = random.sample(memory, bz)
              # Convert the deque to a list to enable slicing
              memory_list = list(memory)
              # === Determine batch selection strategy ===
              use_valid_batch = random.random() < 0.25  # 25% chance to try the valid open???close batch

              if use_valid_batch:
                  # === Your original logic to find a valid open???close batch ===
                  max_retries = 5
                  attempts = 0
                  batch = None

                  preferred_open_action = 1 if self.last_opening_action == 0 else 0

                  while batch is None and attempts < max_retries:
                      attempts += 1
                      start_index = None

                      for i in range(len(memory_list)):
                          action = memory_list[i][2]
                          if action == preferred_open_action:
                              start_index = i
                              self.last_opening_action = preferred_open_action
                              break

                      if start_index is not None:
                          end_index = min(start_index + bz, len(memory_list))
                          batch = memory_list[start_index:end_index]
                          actions = [x[2] for x in batch]

                          if any(a in [5] for a in actions):
                              rewards = [x[3] for x in batch]

                              hold_start = None
                              for i in range(len(actions) - 1, -1, -1):
                                  if actions[i] != 4:
                                      break
                                  hold_start = i

                              if hold_start is not None:
                                  trailing_hold_rewards = rewards[hold_start:]
                                  reward_sum = sum(trailing_hold_rewards)

                                  if reward_sum <= 0 and len(trailing_hold_rewards) >= 3:
                                      print(f"Attempt {attempts}: Trailing hold with no reward (len={len(trailing_hold_rewards)}, reward={reward_sum}) - retrying...")
                                      batch = None
                                      continue
                                  else:
                                      print(f"Attempt {attempts}: Valid open???close batch found.")
                                      self.failed_batch_count = 0
                                      break
                              else:
                                  print(f"Attempt {attempts}: Valid open???close batch found (no hold detected).")
                                  self.failed_batch_count = 0
                                  break
                          else:
                              print(f"Attempt {attempts}: No closing action found - retrying...")
                              batch = None
                      else:
                          print(f"Attempt {attempts}: No valid opening action ({preferred_open_action}) found - retrying...")
                          batch = None

                  # Fallback if validation fails
                  if batch is None:
                      print("Fallback to random batch due to failed valid batch attempts.")
                      batch = stratified_random_batch(memory_list, bz)

              else:
                  # === 75% of the time: train on a random batch immediately ===
                  print("Random Batch Training (75% chance)")
                  batch = stratified_random_batch(memory_list, bz)
              
              if batch:
                batch_hash = hash_batch(batch)
                if batch_hash in trained_batches:
                  print("[SKIP] Batch already trained on - retrying...")
                  batch = None  # force retry/skip
                  return
                else:
                  # Similarity check
                  new_vec = compute_batch_vector(batch).reshape(1, -1)
                  similar = False
                  for past_vec in trained_batches.values():
                      if isinstance(past_vec, str):  # Fix for string case
                        past_vec = np.fromstring(past_vec.strip('[]'), sep=',')
                      past_vec = np.asarray(past_vec).flatten()
                      if len(past_vec) < new_vec.shape[1]:
                        past_vec = np.pad(past_vec, (0, new_vec.shape[1] - len(past_vec)))
                      else:
                        past_vec = past_vec[:new_vec.shape[1]]
                      sim = hybrid_similarity(new_vec.flatten(), past_vec.flatten())
                      if sim > similarity_threshold:
                          print(f"[SKIP] Similar batch found (similarity={sim:.3f}) - skipping.")
                          similar = True
                          batch = None
                          return
        
                  if not similar and batch is not None:
                      print("[ACCEPTED] Batch is unique and dissimilar - training.")
                      trained_batches[batch_hash] = new_vec.flatten()
                      with open(trained_batches_file, "wb") as f:
                          pickle.dump(trained_batches, f)
                  else:
                    print("[SKIP] No valid batch selected - skipping training.")
                    return
              # Create a sequential batch of elements from the memory
              states, original_actions, actions, rewards, next_states, total_rw, timestamp, dones = zip(*batch)
              states = np.vstack(states)
              next_states = np.vstack(next_states)
              rewards = np.array(total_rw)
              reward_mean = np.mean(rewards)
              reward_std = np.std(rewards) + 1e-6  # avoid div-by-zero
              rewards = (rewards - reward_mean) / reward_std
              dones = np.array(dones).astype(int)  # True = 1, False = 0
              
              # Step 1: Predict Q-values from online model (to get best action indices)
              next_q_values_online = self.model.predict(next_states, verbose=0)
              next_actions = np.argmax(next_q_values_online, axis=1)
              
              # Step 2: Predict Q-values from target model (to evaluate selected actions)
              next_q_values_target = self.model2.predict(next_states, verbose=0)
              next_q_selected = next_q_values_target[np.arange(len(next_states)), next_actions]
              
              # Step 3: Compute targets using Double DQN equation
              targets = rewards + self.gamma * next_q_selected * (1 - dones)
              
              # Step 4: Predict current Q-values and apply update to taken actions
              mapped_actions = [inverse_mapping[action] for action in actions]
              target_values = self.model.predict(states, verbose=0)
              target_values[np.arange(len(target_values)), mapped_actions] = targets
              
              # Step 5: Train the model
              self.model.fit(states, target_values, epochs=1, verbose=0)
              
              #if self.epsilon > self.epsilon_min:
                #self.epsilon *= self.epsilon_decay
              return #self.epsilon
            
            else:
              if len(self.memory2) < bz:
                print("not training the model on sequential memory for ... agent-",agent_switch)
                return
              
              print("training the model on sequential memory ", bz ,"... agent-",agent_switch)
              #batch = random.sample(self.memory2, bz)
              # Convert the deque to a list to enable slicing
              memory_list = list(self.memory2)
              # === Determine batch selection strategy ===
              use_valid_batch = random.random() < 0.25  # 25% chance to try the valid open???close batch

              if use_valid_batch:
                  # === Your original logic to find a valid open???close batch ===
                  max_retries = 5
                  attempts = 0
                  batch = None

                  preferred_open_action = 1 if self.last_opening_action == 0 else 0

                  while batch is None and attempts < max_retries:
                      attempts += 1
                      start_index = None

                      for i in range(len(memory_list)):
                          action = memory_list[i][2]
                          if action == preferred_open_action:
                              start_index = i
                              self.last_opening_action = preferred_open_action
                              break

                      if start_index is not None:
                          end_index = min(start_index + bz, len(memory_list))
                          batch = memory_list[start_index:end_index]
                          actions = [x[2] for x in batch]

                          if any(a in [5] for a in actions):
                              rewards = [x[3] for x in batch]

                              hold_start = None
                              for i in range(len(actions) - 1, -1, -1):
                                  if actions[i] != 4:
                                      break
                                  hold_start = i

                              if hold_start is not None:
                                  trailing_hold_rewards = rewards[hold_start:]
                                  reward_sum = sum(trailing_hold_rewards)

                                  if reward_sum <= 0 and len(trailing_hold_rewards) >= 3:
                                      print(f"Attempt {attempts}: Trailing hold with no reward (len={len(trailing_hold_rewards)}, reward={reward_sum}) - retrying...")
                                      batch = None
                                      continue
                                  else:
                                      print(f"Attempt {attempts}: Valid open???close batch found.")
                                      self.failed_batch_count = 0
                                      break
                              else:
                                  print(f"Attempt {attempts}: Valid open???close batch found (no hold detected).")
                                  self.failed_batch_count = 0
                                  break
                          else:
                              print(f"Attempt {attempts}: No closing action found - retrying...")
                              batch = None
                      else:
                          print(f"Attempt {attempts}: No valid opening action ({preferred_open_action}) found - retrying...")
                          batch = None

                  # Fallback if validation fails
                  if batch is None:
                      print("Fallback to random batch due to failed valid batch attempts.")
                      batch = stratified_random_batch(memory_list, bz)

              else:
                  # === 75% of the time: train on a random batch immediately ===
                  print("Random Batch Training (75% chance)")
                  batch = stratified_random_batch(memory_list, bz)
              
              skip_status = False
              if batch:
                batch_hash = hash_batch(batch)
                if batch_hash in trained_batches:
                  print("[SKIP] Batch already trained on - retrying...")
                  batch = None  # force retry/skip
                  skip_status = True
                else:
                  # Similarity check
                  new_vec = compute_batch_vector(batch).reshape(1, -1)
                  similar = False
                  for past_vec in trained_batches.values():
                      if isinstance(past_vec, str):  # Fix for string case
                        past_vec = np.fromstring(past_vec.strip('[]'), sep=',')
                      past_vec = np.asarray(past_vec).flatten()
                      if len(past_vec) < new_vec.shape[1]:
                        past_vec = np.pad(past_vec, (0, new_vec.shape[1] - len(past_vec)))
                      else:
                        past_vec = past_vec[:new_vec.shape[1]]
                      sim = hybrid_similarity(new_vec.flatten(), past_vec.flatten())
                      if sim > similarity_threshold:
                          print(f"[SKIP] Similar batch found (similarity={sim:.3f}) - skipping.")
                          similar = True
                          batch = None
                          skip_status = True
        
                  if not similar and batch is not None:
                      print("[ACCEPTED] Batch is unique and dissimilar - training.")
                      trained_batches[batch_hash] = new_vec.flatten()
                      with open(trained_batches_file, "wb") as f:
                          pickle.dump(trained_batches, f)
                  else:
                    print("[SKIP] No valid batch selected - skipping training.")
                    skip_status = True
              
              if skip_status == False:
                  # Create a sequential batch of elements from the memory
                  states, original_actions, actions, rewards, next_states, total_rw, timestamp, dones = zip(*batch)
                  states = np.vstack(states)
                  next_states = np.vstack(next_states)
                  rewards = np.array(total_rw)
                  reward_mean = np.mean(rewards)
                  reward_std = np.std(rewards) + 1e-6  # avoid div-by-zero
                  rewards = (rewards - reward_mean) / reward_std
                  dones = np.array(dones).astype(int)  # True = 1, False = 0
                  
                  # Step 1: Predict Q-values from online model (to get best action indices)
                  next_q_values_online = self.model2.predict(next_states, verbose=0)
                  next_actions = np.argmax(next_q_values_online, axis=1)
                  
                  # Step 2: Predict Q-values from target model (to evaluate selected actions)
                  next_q_values_target = self.model.predict(next_states, verbose=0)
                  next_q_selected = next_q_values_target[np.arange(len(next_states)), next_actions]
                  
                  # Step 3: Compute targets using Double DQN equation
                  targets = rewards + self.gamma * next_q_selected * (1 - dones)
                  
                  # Step 4: Predict current Q-values and apply update to taken actions
                  mapped_actions = [inverse_mapping[action] for action in actions]
                  target_values = self.model2.predict(states, verbose=0)
                  target_values[np.arange(len(target_values)), mapped_actions] = targets
                  
                  # Step 5: Train the model
                  self.model2.fit(states, target_values, epochs=1, verbose=0)

              mem_bank_dir = "mem_bank"
              mem_files = os.listdir(mem_bank_dir)
    
              if len(mem_files) == 0:
                #if self.epsilon2 > self.epsilon_min:
                #   self.epsilon2 *= self.epsilon_decay
                print("not training the model on memory bank for ... agent-",agent_switch)
                return #self.epsilon2  # No need to train if only one or no memory files
    
              selected_file = random.choice(mem_files)
              file_path = os.path.join(mem_bank_dir, selected_file)
    
              # Load the memory from the selected file
              memory = deque(np.load(file_path, allow_pickle=True), maxlen=max_mem)
              
              print("training the model on memory bank ",bz,"... agent-", agent_switch)
              # Train the model using the loaded memory
              #batch = random.sample(memory, bz)
              # Convert the deque to a list to enable slicing
              memory_list = list(memory)
              # === Determine batch selection strategy ===
              use_valid_batch = random.random() < 0.25  # 25% chance to try the valid open???close batch

              if use_valid_batch:
                  # === Your original logic to find a valid open???close batch ===
                  max_retries = 5
                  attempts = 0
                  batch = None

                  preferred_open_action = 1 if self.last_opening_action == 0 else 0

                  while batch is None and attempts < max_retries:
                      attempts += 1
                      start_index = None

                      for i in range(len(memory_list)):
                          action = memory_list[i][2]
                          if action == preferred_open_action:
                              start_index = i
                              self.last_opening_action = preferred_open_action
                              break

                      if start_index is not None:
                          end_index = min(start_index + bz, len(memory_list))
                          batch = memory_list[start_index:end_index]
                          actions = [x[2] for x in batch]

                          if any(a in [5] for a in actions):
                              rewards = [x[3] for x in batch]

                              hold_start = None
                              for i in range(len(actions) - 1, -1, -1):
                                  if actions[i] != 4:
                                      break
                                  hold_start = i

                              if hold_start is not None:
                                  trailing_hold_rewards = rewards[hold_start:]
                                  reward_sum = sum(trailing_hold_rewards)

                                  if reward_sum <= 0 and len(trailing_hold_rewards) >= 3:
                                      print(f"Attempt {attempts}: Trailing hold with no reward (len={len(trailing_hold_rewards)}, reward={reward_sum}) - retrying...")
                                      batch = None
                                      continue
                                  else:
                                      print(f"Attempt {attempts}: Valid open???close batch found.")
                                      self.failed_batch_count = 0
                                      break
                              else:
                                  print(f"Attempt {attempts}: Valid open???close batch found (no hold detected).")
                                  self.failed_batch_count = 0
                                  break
                          else:
                              print(f"Attempt {attempts}: No closing action found - retrying...")
                              batch = None
                      else:
                          print(f"Attempt {attempts}: No valid opening action ({preferred_open_action}) found - retrying...")
                          batch = None

                  # Fallback if validation fails
                  if batch is None:
                      print("Fallback to random batch due to failed valid batch attempts.")
                      batch = stratified_random_batch(memory_list, bz)

              else:
                  # === 75% of the time: train on a random batch immediately ===
                  print("Random Batch Training (75% chance)")
                  batch = stratified_random_batch(memory_list, bz)
              
              if batch:
                batch_hash = hash_batch(batch)
                if batch_hash in trained_batches:
                  print("[SKIP] Batch already trained on - retrying...")
                  batch = None  # force retry/skip
                  return
                else:
                  # Similarity check
                  new_vec = compute_batch_vector(batch).reshape(1, -1)
                  similar = False
                  for past_vec in trained_batches.values():
                      if isinstance(past_vec, str):  # Fix for string case
                        past_vec = np.fromstring(past_vec.strip('[]'), sep=',')
                      past_vec = np.asarray(past_vec).flatten()
                      if len(past_vec) < new_vec.shape[1]:
                        past_vec = np.pad(past_vec, (0, new_vec.shape[1] - len(past_vec)))
                      else:
                        past_vec = past_vec[:new_vec.shape[1]]
                      sim = hybrid_similarity(new_vec.flatten(), past_vec.flatten())
                      if sim > similarity_threshold:
                          print(f"[SKIP] Similar batch found (similarity={sim:.3f}) - skipping.")
                          similar = True
                          batch = None
                          return
        
                  if not similar and batch is not None:
                      print("[ACCEPTED] Batch is unique and dissimilar - training.")
                      trained_batches[batch_hash] = new_vec.flatten()
                      with open(trained_batches_file, "wb") as f:
                          pickle.dump(trained_batches, f)
                  else:
                    print("[SKIP] No valid batch selected - skipping training.")
                    return
              # Create a sequential batch of elements from the memory
              states, original_actions, actions, rewards, next_states, total_rw, timestamp, dones = zip(*batch)
              states = np.vstack(states)
              next_states = np.vstack(next_states)
              rewards = np.array(total_rw)
              reward_mean = np.mean(rewards)
              reward_std = np.std(rewards) + 1e-6  # avoid div-by-zero
              rewards = (rewards - reward_mean) / reward_std
              dones = np.array(dones).astype(int)  # True = 1, False = 0
              
              # Step 1: Predict Q-values from online model (to get best action indices)
              next_q_values_online = self.model2.predict(next_states, verbose=0)
              next_actions = np.argmax(next_q_values_online, axis=1)
              
              # Step 2: Predict Q-values from target model (to evaluate selected actions)
              next_q_values_target = self.model.predict(next_states, verbose=0)
              next_q_selected = next_q_values_target[np.arange(len(next_states)), next_actions]
              
              # Step 3: Compute targets using Double DQN equation
              targets = rewards + self.gamma * next_q_selected * (1 - dones)
              
              # Step 4: Predict current Q-values and apply update to taken actions
              mapped_actions = [inverse_mapping[action] for action in actions]
              target_values = self.model2.predict(states, verbose=0)
              target_values[np.arange(len(target_values)), mapped_actions] = targets
              
              # Step 5: Train the model
              self.model2.fit(states, target_values, epochs=1, verbose=0)

              #if self.epsilon2 > self.epsilon_min:
              #  self.epsilon2 *= self.epsilon_decay
              return #self.epsilon2
        
    model_filename = 'agent1.h5'  # Define the model file name for saving and loading your trained model
    model_filename2 = 'agent2.h5'  # Define the model file name for saving and loading your trained model
    memory_filename = "memory.npy"
    memory_filename2 = "memory2.npy"
    total_episodes = len(data)  # Set the total number of episodes based on your dataset

    env = TradingEnvironment(data, total_episodes)  # Initialize your custom trading environment with appropriate data
    state_size = len(env.get_state())  # Adjust state size based on your environment's state representation
    action_size = 4  # Set the number of actions according to your trading actions (buy, sell, hold, etc.)

    # Save epi to a file
    save_epi(epi, epi_filename)
    save_rw(reward, rw_filename)
    save_epi2(epi2, epi_filename2)
    save_rw2(reward2, rw_filename2)

    # Load epi from a file if it exists
    epi = load_epi(epi_filename)
    epi2 = load_epi2(epi_filename2)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    csv_filename = f"results/train/train_{timestamp}.csv"
    episode_rewards_df = pd.DataFrame(columns=['Episode', 'Reward', 'Balance'])

    dqn_agent = DQNAgent(state_size, action_size, env, model_filename, model_filename2,memory_filename,memory_filename2)  # Initialize your DQN agent with the appropriate state and action sizes
    
    batch_size = 384  # Adjust the batch size for experience replay
    batch_size1 = 512  # Adjust the batch size for experience replay
    batch_size2 = 640
    batch_size3 = 768
    batch_size4 = 896
    batch_size5 = 1024
    batch_size6 = 1152
    episode_rewards = []
    episode_rewards2 = []
    episode_count = 0
    episode_reward = 0
    episode_reward2 = 0
    episode_balance = 0
    episode_balances = []

    # Load the pre-trained model if it exists
    if os.path.exists(model_filename):
        dqn_agent.model = load_model(model_filename)
        print(f"Loaded model from {model_filename}")
    if os.path.exists(model_filename2):
        dqn_agent.model2 = load_model(model_filename2)
        print(f"Loaded model from {model_filename2}")
    # Load the pre-trained model if it exists
    #if os.path.exists(memory_filename):
    #    dqn_agent.memory = dqn_agent.load_memory(memory_filename)
    #    print(f"Loaded model from {memory_filename}")
    max_retries_per_sample = 5
    retry_counter = 0
    episode_done = False
    
    while episode_count < total_episodes:
        done_loop = False
        failure_done = False
        while not done_loop:
            if env.portfolio['total_equity'] < (10000 * 0.90) or env.portfolio['total_balance'] < (10000 * 0.90):  # Define a custom method in your environment to determine if the episode is done
                if retry_counter > max_retries_per_sample:
                    print(f"Max retries reached ({max_retries_per_sample}). Moving to new sample.")
                    done_loop = True
                    failure_done = True
                    if agent_switch == 1:
                      if 0.1 < epi < 0.5:
                        epi_dif = (0.5 - epi)
                        epi += epi_dif
                        epi += 0.001
                      elif epi < 0.25:
                        epi_dif = (0.25 - epi)
                        epi += epi_dif
                        epi += 0.001
                      else:
                        epi += 0.001
                      epi = max(min(epi, .99), 0.01)
                      save_epi(epi, epi_filename)
                    else:
                      if 0.1 < epi2 < 0.5:
                        epi_dif = (0.5 - epi2)
                        epi2 += epi_dif
                        epi2 += 0.001
                      elif epi2 < 0.25:
                        epi_dif = (0.25 - epi2)
                        epi2 += epi_dif
                        epi2 += 0.001
                      else:
                        epi2 += 0.001
                      epi2 = max(min(epi2, .99), 0.01)
                      save_epi2(epi2, epi_filename2)
                    env.reset() 
                    episode_reward = 0
                    episode_balance = 0
                    episode_count = 0
                    episode_count += 1
                    episode_rewards = []
                    episode_balances = []
                    # Handle episode termination logic if needed
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
                    if reward <= reward2:
                      if reset_p1 > reset_p2 and reset_bias > reset_threshold and trade_bias < reset_threshold:
                        agent_switch = 2
                      else:
                        agent_switch = 1
                    else:
                      if reset_p2 > reset_p1 and reset_bias > reset_threshold and trade_bias < reset_threshold:
                        agent_switch = 1
                      else:
                        agent_switch = 2
                    print("Active agent is, ", agent_switch)
                    print(f"Episode {episode_count} Terminated. Balance is Below Threshold.")
                    retry_counter = 0
                else:
                    print(f"Max retries not reached ({max_retries_per_sample}). replay training from start with same sample.")
                    done_loop = True
                    failure_done = False
                    if agent_switch == 1:
                      if 0.1 < epi < 0.5:
                        epi_dif = (0.5 - epi)
                        epi += epi_dif
                        epi += 0.001
                      elif epi < 0.25:
                        epi_dif = (0.25 - epi)
                        epi += epi_dif
                        epi += 0.001
                      else:
                        epi += 0.001
                      epi = max(min(epi, .99), 0.01)
                      save_epi(epi, epi_filename)
                    else:
                      if 0.1 < epi2 < 0.5:
                        epi_dif = (0.5 - epi2)
                        epi2 += epi_dif
                        epi2 += 0.001
                      elif epi2 < 0.25:
                        epi_dif = (0.25 - epi2)
                        epi2 += epi_dif
                        epi2 += 0.001
                      else:
                        epi2 += 0.001
                      epi2 = max(min(epi2, .99), 0.01)
                      save_epi2(epi2, epi_filename2)
                    env.reset() 
                    episode_reward = 0
                    episode_balance = 0
                    episode_count = 0
                    episode_count += 1
                    episode_rewards = []
                    episode_balances = []
                    # Handle episode termination logic if needed
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
                    if reward <= reward2:
                      if reset_p1 > reset_p2 and reset_bias > reset_threshold and trade_bias < reset_threshold:
                        agent_switch = 2
                      else:
                        agent_switch = 1
                    else:
                      if reset_p2 > reset_p1 and reset_bias > reset_threshold and trade_bias < reset_threshold:
                        agent_switch = 1
                      else:
                        agent_switch = 2
                    print("Active agent is, ", agent_switch)
                    print(f"Episode {episode_count} Terminated. Balance is Below Threshold.")
                    retry_counter += 1
            else:
                state = env.get_state()
                state = np.reshape(state, [1, state_size])
                action, action_prob = dqn_agent.act(state)
                next_state, reward, reward2, total_bal, original_action, action_factor,reset1, reset2, trade_bias, action, total_rw, done = env.step(action, reward, reward2, agent_switch, action_prob)  # Implement the step method in your environment
                next_state = np.reshape(next_state, [1, state_size])
                episode_balance = total_bal
                episode_reward += reward
                episode_reward2 += reward2
                episode_count += 1
                if agent_switch == 1:
                  #if action_factor == -1:
                    #dqn_agent.remember(state, original_action, reward, next_state, done)
                  #else:
                  dqn_agent.remember(state, original_action, action, reward, next_state, total_rw, done)
                else:
                  #if action_factor == -1:
                  #  dqn_agent.remember(state, original_action, reward2, next_state, done)
                  #else:
                  dqn_agent.remember(state, original_action, action, reward2, next_state, total_rw, done)
                env.save_portfolio_normalization_config()
                print(f"Step Reward: {reward}, Cumulative Reward: {episode_reward}")
                print(f"Step Reward2: {reward2}, Cumulative Reward2: {episode_reward2}")
                print(f"Overall Step Reward: {reward+reward2}, Cumulative Reward2: {episode_reward+episode_reward2}")
                print(f"Episode: {episode_count} of Total Episodes: {total_episodes}")
                done_loop = True

        print(f"The Epsilon is {dqn_agent.epsilon}.")
        epi = dqn_agent.epsilon
        print(f"The Epsilon2 is {dqn_agent.epsilon2}.")
        epi2 = dqn_agent.epsilon2
        
        # Append the current step values
        with open("epi_log.csv", mode='a', newline='') as file:
          writer = csv.writer(file)
          writer.writerow([env.current_step, dqn_agent.epsilon, dqn_agent.epsilon2])

        # Visualize episode rewards
        episode_rewards.append(episode_reward+episode_reward2)
        episode_balances.append(episode_balance)
        save_rw(reward, rw_filename)
        save_rw(reward2, rw_filename2)
        episode_rewards_df = episode_rewards_df.append({'Episode': episode_count, 'Reward': reward+reward2, 'Balance': env.portfolio['total_balance']}, ignore_index=True)
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
        if agent_switch == 1:
          print(f"Trained model saved as {model_filename}")
          dqn_agent.model.save(model_filename)
        else:
          print(f"Trained model saved as {model_filename2}")
          dqn_agent.model2.save(model_filename2)

        dqn_agent.save_memory("memory.npy", action)
        dqn_agent.save_memory2("memory2.npy", action)

        # Save epi to a file
        save_epi(epi, epi_filename)
        save_epi2(epi2, epi_filename2)
        save_reset(reset1, reset1_filename)
        save_reset(reset2, reset2_filename)
        print("The active agent is ", agent_switch)
        # Experience replay
        # Load last Q-values from file (e.g., q_values_log.csv)
        # === Track observed min/max globally ===
        if "skip_train_counter" not in globals():
          skip_train_counter = 0
        skip_threshold = 5  # <-- number of skips before forcing training

        if "q_conf_min" not in globals():
            q_conf_min, q_conf_max = float('inf'), float('-inf')
            q_spread_min, q_spread_max = float('inf'), float('-inf')

        q_file = "q_values_select_log.csv"
        if os.path.getsize(q_file) > 0:
            df = pd.read_csv(q_file, header=None)
            last_q_row = ast.literal_eval(df.iloc[-1, 0])
            q_values = last_q_row
            q_confidence = max(q_values)

            if action == 0 or action == 1:
                q_group = [q_values[0], q_values[1]]  # Entry group
            else:
                q_group = [q_values[2], q_values[3]]  # In-position group

            q_spread = max(q_group) - min(q_group)

            # === Update adaptive min/max ===
            q_conf_min = min(q_conf_min, q_confidence)
            q_conf_max = max(q_conf_max, q_confidence)
            q_spread_min = min(q_spread_min, q_spread)
            q_spread_max = max(q_spread_max, q_spread)

            # === Dynamic range protection ===
            conf_range = max(q_conf_max - q_conf_min, 1e-6)
            spread_range = max(q_spread_max - q_spread_min, 1e-6)

            # === Normalize adaptively ===
            conf_scaled = (q_confidence - q_conf_min) / conf_range
            spread_scaled = (q_spread - q_spread_min) / spread_range

        else:
            q_confidence = 0
            q_spread = 0
            conf_scaled = 0
            spread_scaled = 0

        # === Weighted scaling ===
        weight_conf = 0.5
        weight_spread = 0.5
        base_prob = 0.2
        boost_prob = 0.7 * (weight_conf * conf_scaled + weight_spread * spread_scaled)
        train_prob = min(base_prob + boost_prob, 1.0)

        # === Randomly choose to train ===
        should_train = random.random() < train_prob or skip_train_counter >= skip_threshold

        if len(env.positions) <= 0 and should_train:
            dqn_agent.replay(batch_size, batch_size1, batch_size2, batch_size3,
                             batch_size4, batch_size5, batch_size6, agent_switch)
            print(f"[TRAINING] Q-conf: {q_confidence:.3f} | Q-spread: {q_spread:.3f} | prob: {train_prob:.2f}")
        elif len(env.positions) <= 0 and action == 5:
            dqn_agent.replay(batch_size, batch_size1, batch_size2, batch_size3,
                             batch_size4, batch_size5, batch_size6, agent_switch)
            print(f"[TRAINING] End of trade sequence.")
        else:
            print(f"[SKIP] Q-conf: {q_confidence:.3f} | Q-spread: {q_spread:.3f} | prob: {train_prob:.2f}")
        
        if failure_done == True:
            break
          
        if action == 5:
          if len(env.positions) < 1:
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
            if reward <= reward2:
              if reset_p1 > reset_p2 and reset_bias > reset_threshold and trade_bias < reset_threshold:
                agent_switch = 2
              else:
                agent_switch = 1
            else:
              if reset_p2 > reset_p1 and reset_bias > reset_threshold and trade_bias < reset_threshold:
                agent_switch = 1
              else:
                agent_switch = 2
    
        if not env.positions and env.portfolio['total_balance'] < (10000 * 1):  # Check if there are no open positions
            done = True
            if agent_switch == 1:
              # Save epi to a file
              if epi < 0.25:
                  bump = min(0.02, 0.25 - epi)  # softer reset bump
                  epi += bump
              else:
                  epi += 0.01
              epi = max(min(epi, .99), 0.01)
              save_epi(epi, epi_filename)
            else:
              # Save epi to a file
              if epi2 < 0.25:
                bump = min(0.02, 0.25 - epi2)  # softer reset bump
                epi2 += bump
              else:
                epi2 += 0.01
              epi2 = max(min(epi2, .99), 0.01)
              save_epi2(epi2, epi_filename2)
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
            if reward <= reward2:
              if reset_p1 > reset_p2 and reset_bias > reset_threshold and trade_bias < reset_threshold:
                agent_switch = 2
              else:
                agent_switch = 1
            else:
              if reset_p2 > reset_p1 and reset_bias > reset_threshold and trade_bias < reset_threshold:
                agent_switch = 1
              else:
                agent_switch = 2
            print("No open positions and Total Balance less than $10,000. Training terminated.")
            print("Active agent is, ", agent_switch)
            break
        overall_reward = (reward + reward2)/2
        overall_epi = (epi+epi2)/2
        if overall_reward > 0 and overall_epi < .10:
           train, done = test_model(epi,epi2, data, train, reward, reward2, agent_switch)
           if os.path.exists(epi_filename):
             epi = load_epi(epi_filename)
           else:
             epi = 0.5
           print(f"Loaded Epi is {epi}")

           if os.path.exists(epi_filename2):
             epi2 = load_epi2(epi_filename2)
           else:
             epi2 = 0.5
           print(f"Loaded Epi2 is {epi2}")
           break

