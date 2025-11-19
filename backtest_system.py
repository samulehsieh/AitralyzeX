# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 14:44:54 2025

@author: samule
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
import time
import math
import plotly.express as px
import plotly.graph_objects as go
import inspect

# ----------------- 輔助函數 -----------------
# --- 只做字串處理，不抓資料 ---
def format_ticker(t):
    """嘗試用 .TW，若找不到則改用 .TWO"""
    if t.isdigit() and len(t) == 4:
        tw_ticker = t + ".TW"
        data = yf.Ticker(tw_ticker).info
        # yfinance 會在找不到時回傳空字典或沒有 'symbol'
        if not data or "symbol" not in data:
            return t + ".TWO"
        return tw_ticker
    return t

def fetch_data(tickers, period, progress_bar=None, status_text=None, batch_size=50):
    hist_data = {}  # 用來儲存抓取到的資料
    total = len(tickers)
    
    for i in range(0, total, batch_size):
        batch = tickers[i:i + batch_size]
        
        for j, ticker in enumerate(batch):
            if status_text:
                status_text.text(f"正在抓取 {ticker} 的歷史資料… ({i + j + 1}/{total})")
            
            try:
                data = yf.Ticker(ticker).history(period=period)
                if not data.empty:
                    hist_data[ticker] = data
                else:
                    st.warning(f"⚠️ {ticker} 無資料")
            except Exception as e:
                st.warning(f"⚠️ {ticker} 抓取資料失敗: {e}")
            
            if progress_bar:
                progress_bar.progress(min(i + j + 1, total) / total)
            
            time.sleep(0.05)
    
    return hist_data  # 返回抓取到的資料


# === 參數設定 ===
fee_rate = 0.000001425
tax_rate = 0.000003

def price_deviation_breakout_strategy(df, Length=20, Ratio=21):
    df = df.copy()
    
    # 移動平均
    df['MA'] = df['Close'].rolling(window=Length).mean()
    # 偏離條件
    df['Deviation_Condition'] = df['Close'] / df['MA'] <= (1 - Ratio / 100)
    # 當偏離條件成立，記錄當天的高點作為 KPrice
    df['KPrice'] = np.where(df['Deviation_Condition'], df['High'], np.nan)
    # 向前填補空值 (使用最新一個符合條件的高點)
    df['KPrice'] = df['KPrice'].ffill()

    # 進場條件：收盤價突破 KPrice
    df['Cross'] = (df['Close'] > df['KPrice']) & (df['Close'].shift(1) <= df['KPrice'].shift(1))

    # 訊號欄位初始化
    df['Signal'] = 0
    df.loc[df['Cross'], 'Signal'] = 1

    # 出場條件：只要不再符合進場條件（價格未突破 KPrice 且已經持有）
    df['Below_KPrice'] = df['Close'] < df['KPrice']
    df['Exit'] = (df['Below_KPrice']) & (~df['Cross'])  # 當不符合進場且還在持倉

    df.loc[df['Exit'], 'Signal'] = -1

    return df

def bias_crossover_strategy(df, length1=10, length2=20, smooth_length=14, threshold=-2):
    df = df.copy()

    # 計算 BIAS = (收盤 - 均線) / 均線 * 100
    def calc_bias(price, window):
        ma = price.rolling(window=window).mean()
        return (price - ma) / ma * 100

    df["BIAS1"] = calc_bias(df["Close"], length1)
    df["BIAS2"] = calc_bias(df["Close"], length2)
    df["BIAS_DIFF"] = df["BIAS2"] - df["BIAS1"]
    df["BIAS_DIFF_SMOOTH"] = df["BIAS_DIFF"].rolling(window=smooth_length).mean()

    df["Signal"] = 0

    # 偵測穿越 threshold 的進場點
    cond_entry = (df["BIAS_DIFF_SMOOTH"] > threshold) & (df["BIAS_DIFF_SMOOTH"].shift(1) <= threshold)
    df.loc[cond_entry, "Signal"] = 1

    # 出場點為未符合進場條件，即持倉期間若無進場訊號就全部視為出場
    df.loc[(df["Signal"] != 1) & (df["Signal"].shift(1) == 1), "Signal"] = -1

    return df

def pullback_from_high_strategy(df, length=20, percent=7):
    df = df.copy()
    highest_high = df['High'].rolling(window=length).max()
    threshold_price = highest_high * (1 - percent / 100)
    
    df['Signal'] = 0
    df.loc[df['Close'] < threshold_price, 'Signal'] = 1  # 進場條件
    df.loc[df['Close'] >= threshold_price, 'Signal'] = -1  # 出場條件（不再符合進場）

    return df

def simple_crossover_strategy(df):
    df = df.copy()
    df["Signal"] = -1  # 預設為出場

    if len(df) < 7:
        return df

    signals = np.where(
        df["Close"].iloc[6:] > df["Close"].shift(1).iloc[6:], 1, -1
    )
    df.iloc[6:, df.columns.get_loc("Signal")] = signals

    return df

# 計算ATR通道策略
def atr_channel_strategy(df, period=20, N=2):
    df = df.copy()

    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

    df['ATR'] = df['TR'].rolling(window=period).mean()
    df['Middle'] = df['Close'].rolling(window=period).mean()
    df['Upper'] = df['Middle'] + N * df['ATR']
    df['Lower'] = df['Middle'] - N * df['ATR']

    df['Signal'] = 0
    df.loc[(df['Close'] > df['Upper']) & (df['Close'].shift(1) <= df['Upper'].shift(1)), 'Signal'] = 1
    df.loc[(df['Close'] < df['Lower']) & (df['Close'].shift(1) >= df['Lower'].shift(1)), 'Signal'] = -1

    return df

# 計算RSI交叉策略
def rsi_crossover_strategy(df, short_window=6, long_window=12):
    df = df.copy()

    def compute_rsi_wilder(series, window):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def linearregslope(series, window):
        slopes = []
        for i in range(window, len(series) + 1):
            x = np.arange(window)
            y = series.iloc[i-window:i]
            slope, _, _, _, _ = stats.linregress(x, y)
            slopes.append(slope)
        return [np.nan] * (window - 1) + slopes

    df["RSI_Short"] = compute_rsi_wilder(df["Close"], short_window)
    df["RSI_Long"] = compute_rsi_wilder(df["Close"], long_window)
    df["LinearRegSlope_Close"] = linearregslope(df["Close"], 6)
    df["LinearRegSlope_RSI"] = linearregslope(df["RSI_Short"], 6)

    df["Signal"] = 0
    df.loc[(df["LinearRegSlope_Close"] < 0) & (df["LinearRegSlope_RSI"] > 0) &
           (df["Close"] * 1.2 < df["Close"].shift(20)), "Signal"] = 1
    df.loc[(df["RSI_Short"] < df["RSI_Long"]) & (df["RSI_Short"].shift(1) >= df["RSI_Long"].shift(1)), "Signal"] = -1

    return df

def macd_cross_above_zero_strategy(df, fast=12, slow=26, signal=9):
    df = df.copy()

    # 計算 MACD 指標
    df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['DIF'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD'] = df['DIF'].ewm(span=signal, adjust=False).mean()
    df['OSC'] = df['DIF'] - df['MACD']

    # 訊號：OSC（柱狀圖）翻正
    df['Signal'] = np.where((df['OSC'] > 0) & (df['OSC'].shift(1) <= 0), 1, 0)

    # 出場訊號：昨天進場但今天沒有新的進場訊號
    df['Signal'] = np.where((df['Signal'] != 1) & (df['Signal'].shift(1) == 1), -1, df['Signal'])

    return df

def momentum_cross_above_zero_strategy(df, length=10):
    df = df.copy()
    
    # 計算 Momentum 指標
    df['Momentum'] = df['Close'] - df['Close'].shift(length)
    
    # 進場條件：Momentum 從小於等於 0 翻正
    df['Signal'] = np.where((df['Momentum'] > 0) & (df['Momentum'].shift(1) <= 0), 1, 0)
    
    # 出場條件：昨天是進場訊號，今天不是
    df['Signal'] = np.where((df['Signal'] != 1) & (df['Signal'].shift(1) == 1), -1, df['Signal'])

    return df

def bollinger_oversold_strategy(df, length=20, lower_band=2):
    df = df.copy()
    
    # 計算布林通道下緣
    ma = df['Close'].rolling(window=length).mean()
    std = df['Close'].rolling(window=length).std()
    lower = ma - lower_band * std

    # 進場條件：最低價 <= 下緣 → 超賣進場
    df['Signal'] = np.where(df['Low'] <= lower, 1, -1)

    # 只在今天是買進、昨天不是的情況下給 1 訊號（防止持續買進）
    df['Signal'] = np.where(
        (df['Signal'] == 1) & (df['Signal'].shift(1) != 1),
        1,
        np.where(
            (df['Signal'] == -1) & (df['Signal'].shift(1) != -1),
            -1,
            0
        )
    )

    return df

def dual_ma_crossover_strategy(df, short_length=5, long_length=20, price_col='Close'):
    df = df.copy()
    
    df['ma_short'] = df[price_col].rolling(window=short_length).mean()
    df['ma_long'] = df[price_col].rolling(window=long_length).mean()

    # 訊號產生：短均線上穿長均線 = 買；否則賣出
    df['Signal'] = np.where(
        (df['ma_short'] > df['ma_long']) & (df['ma_short'].shift(1) <= df['ma_long'].shift(1)),
        1,
        -1
    )

    # 清除重複訊號
    df['Signal'] = np.where(
        (df['Signal'] == 1) & (df['Signal'].shift(1) != 1),
        1,
        np.where(
            (df['Signal'] == -1) & (df['Signal'].shift(1) != -1),
            -1,
            0
        )
    )

    return df

def backtest(df, initial_capital, fee_rate, tax_rate):
    df = df.copy()
    df = df.reset_index()
    df['買點1'] = 0
    df['買點2'] = None
    df['賣點'] = None
    holding = False

    # 幫助函數：判斷今天的有效進場訊號
    def get_valid_today_signal(signal_series):
        last_nonzero_idx = None
        last_nonzero_value = 0

        # 從倒數第二筆往前找最後一次非0訊號
        for i in range(len(signal_series) - 2, -1, -1):
            if signal_series.iloc[i] != 0:
                last_nonzero_idx = i
                last_nonzero_value = signal_series.iloc[i]
                break

        today_signal = signal_series.iloc[-1]

        # 今天是買進訊號，且前一次非0訊號是賣出訊號，視為有效進場
        if today_signal == 1 and last_nonzero_value == -1:
            return 1
        else:
            return 0

    for i in range(1, len(df)):
        if df.at[i, "Signal"] == 1:
            df.at[i, '買點1'] = 1
        else:
            df.at[i, '買點1'] = 0

        if df.at[i, '買點1'] == 1 and df.at[i - 1, '買點1'] != 1 and not holding:
            df.at[i, '買點2'] = 1
            holding = True

        if df.at[i, "Signal"] == -1 and holding:
            df.at[i, '賣點'] = 1
            holding = False

    if holding:
        df.at[len(df) - 2, '賣點'] = 1

    buy_points = df[df['買點2'].notnull()].index
    sell_points = df[df['賣點'].notnull()].index
    sell_idx_pointer = 0
    capital = initial_capital

    trades = []
    equity_curve = [capital]

    for buy_idx in buy_points:
        if buy_idx + 1 < len(df):
            buy_date = df.at[buy_idx + 1, 'Date']
        else:
            continue
        buy_price = df.at[buy_idx + 1, 'Open']

        if capital < buy_price:
            continue

        for j in range(sell_idx_pointer, len(sell_points)):
            sell_idx = sell_points[j]
            sell_date = df.at[sell_idx + 1, 'Date']
            sell_price = df.at[sell_idx + 1, 'Open'] + df.loc[buy_idx:sell_idx + 1, 'Dividends'].sum()

            if sell_date > buy_date:
                ret = (sell_price - buy_price) / buy_price
                fee = (buy_price * fee_rate) + (sell_price * fee_rate)
                tax = sell_price * tax_rate
                total_cost = fee + tax
                profit = sell_price - buy_price - total_cost
                capital += profit
                holding_days = (sell_date - buy_date).days

                trades.append({
                    '進場日期': buy_date,
                    '進場價格': buy_price,
                    '出場日期': sell_date,
                    '出場價格': sell_price,
                    '報酬率.': ret,
                    '報酬率': f"{ret * 100:.2f}%",
                    '手續費': fee,
                    '稅費': tax,
                    '淨利': profit*1000,
                    '持有天數': holding_days
                })

                equity_curve.append(capital)
                sell_idx_pointer = j + 1
                break

    if len(trades) == 0:
        return None, None

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['進場日期'] = pd.to_datetime(trades_df['進場日期']).dt.normalize()
    trade_returns = trades_df['報酬率.'].values
    trade_profits = trades_df['淨利'].values
    fees = trades_df['手續費'].values
    taxes = trades_df['稅費'].values

    win_mask = trade_profits > 0
    loss_mask = trade_profits < 0

    gross_profit = trade_profits[win_mask].sum()
    gross_loss = trade_profits[loss_mask].sum()

    # 取每筆交易期間的價格序列計算單張回撤
    all_drawdowns = []
    for buy_idx, sell_idx in zip(buy_points, sell_points):
        prices = df.loc[buy_idx:sell_idx, 'Close'].values
        running_max = np.maximum.accumulate(prices)
        drawdowns = (prices - running_max) / running_max
        if len(drawdowns) == 0:
            all_drawdowns.append(0)
        else:
            all_drawdowns.append(drawdowns.min())

    
    max_drawdown_ratio = min(all_drawdowns)



    max_trade_loss = trade_profits.min()
    std = trade_returns.std()
    downside_std = trade_returns[trade_returns < 0].std() if np.any(trade_returns < 0) else 0
    buy_and_hold_return = (df['Close'].iloc[-1] + df['Dividends'].sum() - df['Open'].iloc[0]) / initial_capital
    today_signal = get_valid_today_signal(df['Signal'])

    metrics = {
        '初始資金': f"{initial_capital * 1000:.3f}元",
        '最終權益': f"{capital * 1000:.3f}元",
        '總報酬率': f"{((capital / initial_capital) -1)*100:.2f}%",
        '買入持有報酬率': f"{(buy_and_hold_return)*100:.2f}%",
        '策略相對買入持有超額報酬': f"{(((capital / initial_capital) -1)-buy_and_hold_return)*100:.2f}%",
        '總交易次數': f"{len(trades)}次",
        #'總手續費成本': f"{fees.sum()*1000:.3f}元",
        #'總證交稅成本': f"{taxes.sum()*1000:.3f}元",
        #'平均每筆手續費': f"{fees.mean()*1000:.3f}元",
        #'獲利交易次數': f"{win_mask.sum()}次",
        #'虧損交易次數': f"{loss_mask.sum()}次",
        '勝率': f"{(win_mask.sum() / len(trades))*100:.2f}%",
        #'平均獲利金額': f"{(trade_profits[win_mask].mean() if win_mask.any() else 0)*1000:.3f}元",
        #'平均虧損金額': f"{(trade_profits[loss_mask].mean() if loss_mask.any() else 0)*1000:.3f}元",
        '盈虧比': f"{(trade_profits[win_mask].mean() / abs(trade_profits[loss_mask].mean())) if win_mask.any() and loss_mask.any() else 0:.3f}",
        '盈利因子': f"{(gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf:.3f}",
        '標準差': f"{(std)*100:.2f}%",
        '最大回撤': f"{(max_drawdown_ratio)*100:.2f}%",
        #'最大交易虧損': f"{max_trade_loss*1000:.3f}元",
        '夏普比率': f"{(trade_returns.mean() / std * np.sqrt(252)) if std != 0 else 0:.3f}",
        '索提諾比率': f"{(trade_returns.mean() / downside_std * np.sqrt(252)) if downside_std != 0 else 0:.3f}",
        '平均持有天數': f"{trades_df['持有天數'].mean():.3f}天",
        '今天的訊號': today_signal
    }

    return metrics, trades

# --- 策略定義區 ---
strategies = [
    ("價格偏離突破策略", price_deviation_breakout_strategy),
    ("BIAS差值穿越策略", bias_crossover_strategy),
    ("高點回檔策略", pullback_from_high_strategy),
    ("簡單突破策略", simple_crossover_strategy),
    ("ATR通道突破策略", atr_channel_strategy),
    ("RSI越線策略", rsi_crossover_strategy),
    ("MACD柱狀圖翻正策略", macd_cross_above_zero_strategy),
    ("Momentum翻正策略", momentum_cross_above_zero_strategy),
    ("Bollinger超賣策略", bollinger_oversold_strategy),
    ("雙均線交叉策略", dual_ma_crossover_strategy)
]
strategy_object_map = {name: func for name, func in strategies}

strategy_names = [s[0] for s in strategies]
def toggle_select_all(page_prefix, strategy_names):
    sel_key = f"{page_prefix}_selected_strategies"
    all_key = f"{page_prefix}_select_all"
    multi_key = f"{page_prefix}_multiselect_key"

    if multi_key not in st.session_state:
        st.session_state[multi_key] = 0

    if st.session_state[all_key]:
        st.session_state[sel_key] = strategy_names.copy()
    else:
        st.session_state[sel_key] = []

    # 🔹 強制刷新 multiselect
    st.session_state[multi_key] += 1

        
def sync_select_all(page_prefix, strategy_names):
    key_sel = f"{page_prefix}_selected_strategies"
    key_all = f"{page_prefix}_select_all"
    st.session_state[key_all] = len(st.session_state[key_sel]) == len(strategy_names)

# --- 單策略參數調整 ---
strategy_params = {
    "價格偏離突破策略": {"Length": 20, "Ratio": 21},
    "BIAS差值穿越策略": {"length1": 10, "length2": 20, "smooth_length": 14, "threshold": -2},
    "高點回檔策略": {"length": 20, "percent": 7},
    "ATR通道突破策略": {"period": 20, "N": 2},
    "RSI越線策略": {"short_window": 6, "long_window": 12},
    "MACD柱狀圖翻正策略": {"fast": 12, "slow": 26, "signal": 9},
    "Momentum翻正策略": {"length": 10},
    "Bollinger超賣策略": {"length": 20, "lower_band": 2},
    "雙均線交叉策略": {"short_length": 5, "long_length": 20},
}

def update_selected_strategies(page_prefix, strategy_names):
    sel_key = f"{page_prefix}_selected_strategies"
    all_key = f"{page_prefix}_select_all"
    # 更新 checkbox 狀態
    st.session_state[all_key] = len(st.session_state[sel_key]) == len(strategy_names)

strategy_descriptions = {
    "價格偏離突破策略":"概念：偵測股價偏離平均線，尋找短期反彈<br>""進場：價格突破近期高點時買進<br>""出場：價格回落時賣出<br>""特點：偏向捕捉短期反彈",
    "BIAS差值穿越策略":"概念：利用短期與中期 BIAS（乖離率）之差值，並搭配平滑指標，偵測由弱轉強的反轉訊號<br>""進場：平滑後的 BIAS 差值向上突破設定 threshold 時買進（由低於門檻 → 高於門檻）<br>""出場：當 BIAS 差值不再維持進場條件時視為出場（指標跌回 threshold 下方）<br>""特點：屬於反轉型策略，偏向捕捉乖離修正後的向上轉折",
    "高點回檔策略":"概念：利用近期高點的回檔幅度偵測超跌狀態，尋找可能的反彈買點<br>""進場：收盤價跌破過去 length 期高點的 percent% 下方時買進（出現明顯回檔）<br>""出場：收盤價重新站回 threshold 以上時賣出（不再處於深度回檔區）<br>""特點：屬於逆勢逢低承接策略，適合震盪或急跌後反彈行情",
    "簡單突破策略":"概念：以最基本的收盤價突破前一日收盤價作為判斷依據，捕捉短線方向性變化<br>""進場：收盤價高於前一天收盤價時買進（出現向上突破）<br>""出場：收盤價低於前一天收盤價時賣出（向下跌破）<br>""特點：極為簡單的動能判斷策略，反應快速但可能受到雜訊影響",
    "ATR通道突破策略":"概念：利用 ATR 建構上下通道，透過價格突破波動區間偵測強勢方向的產生<br>""進場：收盤價突破上軌（Upper）且前一天未突破時買進，代表向上脫離震盪區<br>""出場：收盤價跌破下軌（Lower）且前一天未跌破時賣出，代表向下跌穿支撐區<br>""特點：屬於波動通道突破策略，能有效捕捉高波動突破行情，適合順勢交易",
    "RSI越線策略":"概念：結合短期與長期 RSI 以及線性回歸斜率，偵測股價動能反轉訊號<br>""進場：當股價短期下跌趨勢（LinearRegSlope_Close < 0）且短期 RSI 斜率向上（LinearRegSlope_RSI > 0），且股價低於 20 期前的 1.2 倍時買進<br>""出場：短期 RSI 自上向下跌破長期 RSI 時賣出（RSI_Short < RSI_Long 且昨日 RSI_Short >= RSI_Long）<br>""特點：屬於動能反轉策略，結合 RSI 趨勢與價格斜率，用於捕捉超跌後的反彈機會",
    "MACD柱狀圖翻正策略":"概念：利用 MACD 柱狀圖從負轉正的訊號，捕捉中短期多頭反轉起點<br>""進場：當 MACD OSC（柱狀圖）由負轉正時買進<br>""出場：若前一天持倉但今天沒有新的翻正訊號則賣出<br>""特點：適合捕捉中短線反彈或趨勢轉折點，操作相對頻繁，信號明確",
    "Momentum翻正策略":"概念：利用 Momentum 指標偵測股價動能從負轉正的起點，捕捉短中期反彈<br>""進場：當 Momentum 指標由小於等於 0 翻正時買進<br>""出場：若前一天持倉但今天沒有新的翻正訊號則賣出<br>""特點：偏向捕捉短中期反彈，操作頻率中等，信號清晰且容易理解",
    "Bollinger超賣策略":"概念：利用布林通道下緣判斷股價是否超賣<br>""進場：當股價最低價觸及布林通道下緣時買進<br>""出場：若前一天持倉但今天股價未再觸及下緣則賣出<br>""特點：偏向捕捉短期反彈，適合震盪或回檔行情",
    "雙均線交叉策略":"概念：利用短期均線與長期均線的交叉判斷買賣訊號。<br>""進場：短期均線由下向上穿過長期均線 → 買進<br>""出場：短期均線由上向下穿過長期均線 → 賣出<br>""特點：經典趨勢追蹤策略，適合捕捉中短期趨勢"
}


# --- 在回測執行區域前加上這段邏輯 ---
def load_hist_data(formatted_tickers, period_option, progress_bar, status_text):
    """檢查快取的 hist_data 是否可重用，否則重新抓取"""
    # 判斷是否已有資料且相同條件
    if (
        "hist_data" in st.session_state and
        st.session_state.hist_data is not None and
        st.session_state.hist_data_tickers == formatted_tickers and
        st.session_state.hist_data_period == period_option
    ):
        return st.session_state.hist_data  # ✅ 直接重用
    else:
        # 🚀 若無快取或條件不同，重新抓取並快取起來
        hist_data = fetch_data(formatted_tickers, period_option, progress_bar, status_text)
        st.session_state.hist_data = hist_data
        st.session_state.hist_data_tickers = formatted_tickers
        st.session_state.hist_data_period = period_option
        return hist_data

def format_ticker_to_name(ticker_code_tw, stock_dict):
    """將 '代碼.TW' 轉換為 '代碼 名稱'"""
    # 1. 移除 .TW 或 .TWO (或其他)
    base_code = ticker_code_tw.split('.')[0]
    
    # 2. 查找名稱
    stock_name = stock_dict.get(base_code, "名稱未知")
    
    # 3. 組合成所需格式
    return f"{base_code} {stock_name}"

# 確保 stock_dict 在全域或 with tab_signal 內可被訪問
# 假設 stock_dict 已經在 Streamlit 腳本開頭被正確定義

# ----------------- LLM Agent Configuration -----------------
try:
    # 確保您已安裝 Google Generative AI SDK: pip install google-genai
    from google import genai 
    
    # 使用 Streamlit secrets 取得 API Key (慣用做法)
    if "gemini_api_key" in st.secrets:
        # 這是真正的 Client 初始化，使用 st.secrets 取得的 API Key
        client = genai.Client(api_key=st.secrets["gemini_api_key"])
    else:
        st.error("LLM 服務未啟用：請在 Streamlit 的 secrets 中設定 gemini_api_key。")
        client = None

except ImportError:
    st.error("LLM 服務未啟用：請安裝 Google Generative AI SDK (pip install google-genai)。")
    client = None
except Exception as e:
    st.error(f"LLM Client 初始化時發生錯誤: {e}")
    client = None

def llm_api_call(prompt_text):
    
    if client is None:
        return "LLM 服務尚未啟用。請檢查 API Key 設定和函式庫安裝。"

    try:
        # 這是真正的 LLM API 呼叫 (使用 gemini-2.5-flash 模型)
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt_text
        )
        return response.text
        
    except Exception as e:
        return f"LLM 服務呼叫失敗。錯誤資訊：{e}"
# ----------------- LLM Agent Configuration End -----------------
    
import streamlit as st

# ----------------- Streamlit 介面 -----------------
st.title("股票及時訊號與策略回測")
# ----------------- 分頁 -----------------
tab_signal, tab_backtest = st.tabs(["訊號產出", "回測結果"])

# ----------------- 訊號產出頁 -----------------
with tab_signal:
    st.subheader("股票訊號產出")

    page_prefix = "signal"

    # --- 初始化 session_state ---
    default_keys = {
        "signal_ticker_list": [],
        "signal_new_ticker": "",
        "signal_is_running": False,
        "signal_stop_requested": False,
        "signal_selected_strategies": [],
        "signal_multiselect_key": 0,
        "signal_df_results": pd.DataFrame(),
        "signal_df_trades": pd.DataFrame(),
    }
    for key, default in default_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default

    for key in [f"{page_prefix}_selected_strategies", f"{page_prefix}_select_all", f"{page_prefix}_multiselect_key"]:
        if key not in st.session_state:
            if "strategies" in key:
                st.session_state[key] = []
            elif "multiselect_key" in key:
                st.session_state[key] = 0
            else:
                st.session_state[key] = False

    # --- 讀取股票池 CSV ---
    url = "https://raw.githubusercontent.com/samulehsieh/-/refs/heads/main/%E8%82%A1%E7%A5%A8%E6%B1%A0.csv"
    df = pd.read_csv(url, dtype=str, encoding='utf-8-sig')
    stock_dict = dict(zip(df['代號'], df['名稱']))
    options = [f"{code} {name}" for code, name in stock_dict.items()]

    
    # 🔹 所有輸入欄位直接綁定 state
    selected_from_search = st.multiselect(
        "搜尋股票名稱或代號（點選標籤即可選取）",
        options=options,
        default=[f"{code} {stock_dict[code]}" for code in st.session_state.signal_ticker_list],
        disabled=st.session_state.signal_is_running,  # 🔹 根據 is_running 凍結
        key=f"signal_stock_selector_{page_prefix}_{st.session_state.signal_multiselect_key}"
    )
    

    if selected_from_search != [f"{code} {stock_dict[code]}" for code in st.session_state.signal_ticker_list]:
        st.session_state.signal_ticker_list = [item.split()[0] for item in selected_from_search]

    tickers = st.session_state.signal_ticker_list[:]
    st.write(f"股票代碼清單：{tickers}{'...' if len(st.session_state.signal_ticker_list) > 20 else ''} （共 {len(st.session_state.signal_ticker_list)} 支）")

    # ----------------- 策略多選框 -----------------
    st.checkbox(
        "全選策略",
        key=f"{page_prefix}_select_all",
        on_change=toggle_select_all,
        args=(page_prefix, strategy_names),
        disabled=st.session_state.signal_is_running  # 🔹 查詢時凍結
    )
    
    st.multiselect(
        "選擇策略",
        options=strategy_names,
        key=f"{page_prefix}_selected_strategies",
        default=st.session_state[f"{page_prefix}_selected_strategies"],
        on_change=sync_select_all,
        args=(page_prefix, strategy_names),
        disabled=st.session_state.signal_is_running  # 🔹 查詢時凍結
    )

    filtered_strategies = [
        s for s in strategies if s[0] in st.session_state[f"{page_prefix}_selected_strategies"]
    ]

    # ----------------- 顯示單策略參數 -----------------
    params = {}
    if len(filtered_strategies) == 1:
        selected_strategy_name, _ = filtered_strategies[0]
        if selected_strategy_name in strategy_params:
            st.subheader(f"{selected_strategy_name} 參數設定")
            for param_name, default_value in strategy_params[selected_strategy_name].items():
                if isinstance(default_value, int):
                    params[param_name] = st.number_input(
                        param_name,
                        value=default_value,
                        #disabled=inputs_disabled
                    )
                elif isinstance(default_value, float):
                    params[param_name] = st.number_input(
                        param_name,
                        value=default_value,
                        format="%.4f",
                        #disabled=inputs_disabled
                    )

    # ----------------- 回測控制區 -----------------
    col1, col2 = st.columns(2)
    
    # 顯示回測控制按鈕
    with col1:
        if st.session_state.signal_is_running:
            if st.button("⏹ 停止查詢", key="stop_button_signal"):
                st.session_state.signal_stop_requested = True
                st.session_state.signal_is_running = False
                st.rerun()
        else:
            if st.button("▶️ 開始查詢", key="start_button_signal"):
                st.session_state.signal_is_running = True
                st.session_state.signal_stop_requested = False
                if not st.session_state.signal_df_results.empty:
                    st.session_state.signal_df_results = pd.DataFrame()
                    st.session_state.signal_df_trades = pd.DataFrame()
                st.session_state.signal_multiselect_key += 1
                st.rerun()
    
   # --- 回測執行邏輯 ---
    if st.session_state.signal_is_running:
        if not tickers:
            st.warning("⚠️ 請先加入至少一支股票！")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
    
            # 格式化 tickers 並過濾掉空的值
            formatted_tickers = [ft for ft in (format_ticker(c) for c in tickers) if ft]
            status_text.text("正在抓取資料中...")
            
            # 呼叫 fetch_data 抓取歷史資料
            hist_data = load_hist_data(formatted_tickers, "2y", progress_bar, status_text)
    
            # 檢查 hist_data 是否是字典，並且不為空
            if not isinstance(hist_data, dict) or not hist_data:
                st.error("⚠️ 無法獲取資料或資料格式錯誤。請檢查股票代碼或資料源。")
            else:
                total_tasks = len(hist_data) * len(filtered_strategies)
                task_count = 0
                results = []
    
                fee_rate = 0.000001425
                tax_rate = 0.000003
    
                # 開始回測
                for ticker, hist in hist_data.items():
                    for strategy_name, strategy_func in filtered_strategies:
                        if st.session_state.signal_stop_requested:
                            st.warning("⏹ 已手動停止回測")
                            st.session_state.signal_is_running = False
                            break
    
                        status_text.text(f"正在回測 {ticker} - {strategy_name} ({task_count+1}/{total_tasks})")
    
                        try:
                            # 確保策略函數可用
                            if len(filtered_strategies) == 1 and strategy_name in strategy_params:
                                df_signal = strategy_func(hist, **params)
                            else:
                                df_signal = strategy_func(hist)
                            
                            metrics, trades = backtest(df_signal, 10000000, fee_rate, tax_rate)
    
                            if metrics:
                                # 創建一個新的有序字典 (ordered_metrics)，將關鍵欄位放在最前面
                                ordered_metrics = {
                                    "股票代碼": ticker,
                                    "策略名稱": strategy_name,
                                    **metrics # 將 backtest 函數返回的所有其他指標解包放在後面
                                }
                                results.append(ordered_metrics)
                            
                                # 保存每筆交易明細
                                if 'df_trades' not in st.session_state:
                                    st.session_state.signal_df_trades = []
                                for t in trades:
                                    t["策略名稱"] = strategy_name
                                    t["股票代碼"] = ticker
                                    st.session_state.signal_df_trades.append(t)
                            
                            # 最後將交易明細轉成 DataFrame
                            if 'df_trades' in st.session_state:
                                st.session_state.signal_df_trades = pd.DataFrame(st.session_state.signal_df_trades)
    
                        except Exception as e:
                            pass
    
    
                        task_count += 1
                        progress_bar.progress(task_count / total_tasks)
    
                # 回測完成，顯示結果並讓按鈕恢復為「開始回測」
                if not st.session_state.signal_stop_requested:
                    st.session_state.signal_is_running = False
                    st.success("✅ 回測完成！")
                    st.session_state.signal_df_results = pd.DataFrame(results)  # 保存結果至 session_state
                    st.rerun()  # 重新渲染頁面，更新按鈕狀態
    
    # 篩選條件
    if 'signal_df_results' in st.session_state and not st.session_state.signal_df_results.empty:
        df_results = st.session_state.signal_df_results
    
        # 如果 df_results 存在，並且 "今天的訊號" 有值，進行數據類型轉換
        df_results["今天的訊號"] = df_results["今天的訊號"].astype("float").fillna(0).astype("int")
    
        # 確保 formatted_tickers 被定義
        formatted_tickers = [ft for ft in (format_ticker(c) for c in tickers) if ft]
        
        # 在篩選邏輯中初始化 progress_bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        hist_data = st.session_state.get("hist_data", None)
        if hist_data is None:
            st.warning("⚠️ 尚未載入歷史資料，請先執行回測。")
            st.stop()
        
        if not hist_data or not isinstance(hist_data, dict) or len(hist_data) == 0:
            st.error("⚠️ 無法獲取資料或資料格式錯誤，請檢查資料來源或股票代碼。")
        else:
            # 進行篩選和回測邏輯
            filtered_results = []
            for ticker, hist in hist_data.items():
                df_ticker = df_results[df_results["股票代碼"] == ticker]
                df_ticker_filtered = df_ticker[
                    (df_ticker["今天的訊號"] == 1)
                ]
                filtered_results.append(df_ticker_filtered)
        
            # 將篩選結果合併成一個 DataFrame
            filtered_df = pd.concat(filtered_results, ignore_index=True) if filtered_results else pd.DataFrame()
            if filtered_df.empty:
                st.info("❗ 沒有符合條件的策略回測結果。")
            else:
                # 1. 應用股票名稱轉換 (使用您剛才的邏輯)
                #    確保 stock_dict 在此處可用
                filtered_df['股票'] = filtered_df['股票代碼'].apply(
                    lambda x: format_ticker_to_name(x, stock_dict)
                )
            
                # 2. 僅選擇 '股票' 和 '策略名稱' 這兩個欄位
                final_display_df = filtered_df[['股票', '策略名稱']]
                
                # 3. 顯示結果
                st.subheader("🎯 符合條件的訊號清單")
                st.dataframe(final_display_df, hide_index=True) # 隱藏 DataFrame 索引讓它看起來更像一個清單

with tab_backtest:
    st.subheader("股票策略回測")

    page_prefix = "backtest"

    # --- 初始化 session_state ---
    default_keys = {
        "backtest_ticker_list": [],
        "backtest_new_ticker": "",
        "backtest_is_running": False,
        "backtest_stop_requested": False,
        "backtest_selected_strategies": [],
        "backtest_multiselect_key": 0,
        "backtest_results": {},         
        "backtest_trades": {},          
        "backtest_params": {},          
        "backtest_hist_data": {},
    }
    for key, default in default_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # --- 股票池 ---
    url = "https://raw.githubusercontent.com/samulehsieh/-/refs/heads/main/%E8%82%A1%E7%A5%A8%E6%B1%A0.csv"
    df = pd.read_csv(url, dtype=str, encoding='utf-8-sig')
    stock_dict = dict(zip(df['代號'], df['名稱']))

    options = [f"{code} {name}" for code, name in stock_dict.items()]

    # --- 確保 ticker 初始值 ---
    if not st.session_state.backtest_ticker_list:
        first_code = options[0].split()[0]
        st.session_state.backtest_ticker_list = [first_code]

    default_fullname = f"{st.session_state.backtest_ticker_list[0]} {stock_dict[st.session_state.backtest_ticker_list[0]]}"
    default_index = options.index(default_fullname) if default_fullname in options else 0

    # --- 股票 selectbox（單選） ---
    selected_from_search = st.selectbox(
        "搜尋股票名稱或代號",
        options=options,
        index=default_index,
        disabled=st.session_state.backtest_is_running,
        key=f"stock_selector_{st.session_state.backtest_multiselect_key}"
    )

    selected_code = selected_from_search.split()[0]
    st.session_state.backtest_ticker_list = [selected_code]

    tickers = st.session_state.backtest_ticker_list[:]
    st.write(f"股票代碼清單：{tickers}")

    # --- 初始資金 ---
    initial_capital = st.number_input(
        "初始資金(元)", min_value=1000, value=100000, step=1000,
        disabled=st.session_state.backtest_is_running
    ) / 1000

    # ----------------- 策略選擇（單選） -----------------
    strategy_state_key = f"{page_prefix}_selected_strategies"

    # 初始化策略狀態
    if not st.session_state[strategy_state_key]:
        st.session_state[strategy_state_key] = [strategy_names[0]]

    current_strategy = st.session_state[strategy_state_key][0]
    default_strategy_index = (
        strategy_names.index(current_strategy)
        if current_strategy in strategy_names else 0
    )

    selected_strategy = st.selectbox(
        "選擇策略",
        options=strategy_names,
        index=default_strategy_index,
        key=f"{page_prefix}_selected_strategy",
        disabled=st.session_state.backtest_is_running
    )

    st.session_state[strategy_state_key] = [selected_strategy]

    chat_context_key = "chat_strategy_context" # 用於記錄當前聊天記錄屬於哪個策略

    # 檢查當前選定的策略是否與聊天記錄中記錄的策略一致。
    if chat_context_key not in st.session_state:
        # 第一次運行時初始化上下文
        st.session_state[chat_context_key] = selected_strategy
        
    elif st.session_state[chat_context_key] != selected_strategy:
        # 發現策略已切換 (即：新策略名稱 != 舊策略名稱)
        
        # 1. 清空聊天記錄
        if "messages" in st.session_state:
            st.session_state.messages = [] 
        
        # 2. 更新上下文標記為新策略名稱
        st.session_state[chat_context_key] = selected_strategy

    filtered_strategies = [
        s for s in strategies
        if s[0] in st.session_state[strategy_state_key]
    ]

    # ----------------- 回測控制按鈕 -----------------
    col1, col2 = st.columns(2)

    # 安全初始化（保險）
    if "backtest_params" not in st.session_state:
        st.session_state.backtest_params = {}
    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = {}
    if "backtest_trades" not in st.session_state:
        st.session_state.backtest_trades = {}

    with col1:
        if st.session_state.backtest_is_running:
            if st.button("⏹ 停止回測"):
                st.session_state.backtest_stop_requested = True
                st.session_state.backtest_is_running = False
                st.rerun()
        else:
            if st.button("▶️ 開始回測"):
                st.session_state.backtest_is_running = True
                st.session_state.backtest_stop_requested = False

                # 🔥 清空先前資料
                st.session_state.backtest_results = {}
                st.session_state.backtest_trades = {}
                st.rerun()

    st.write("---")

    
   
    # ======================================================
    # 狀態佔位符 (必須放在分頁定義之上，確保顯示在應用程式頂部)
    # ======================================================
    status_container = st.empty() 
    
    # ======================================================
    # 第二層分頁
    # ======================================================
    subtab_intro, subtab_perf, subtab_params = st.tabs(
        ["策略說明", "績效報告", "參數調整"]
    )
    
    # ----------------- 策略說明 -----------------
    with subtab_intro:
    
        # 確保 current_strategy 能夠被獲取
        current_strategy = st.session_state.get("backtest_selected_strategies", ["預設策略"])[0]
        
        st.markdown(f"### 📌 {current_strategy}")
    
        description = strategy_descriptions.get(
            current_strategy,
            "此策略尚未提供說明。"
        )
        st.markdown(description, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("🤖 策略問答助手 (LLM Chat Agent)")
        st.caption("您可以連續提問，AI 會記住之前的對話內容。AI 的所有解釋都將基於程式碼。")
    
        # =====================================================
        # LLM 應用：多輪對話 RAG Agent
        # =====================================================
        
        # 1. 初始化聊天記錄
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
    
        # 2. 顯示歷史聊天記錄
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
    
        # 3. 處理使用者輸入
        if prompt := st.chat_input("請輸入您的策略相關問題...", key="main_chat_input"):
            
            # 將使用者問題添加到聊天記錄
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 顯示最新的使用者問題
            with st.chat_message("user"):
                st.markdown(prompt)
    
            # 準備 AI 回答
            with st.chat_message("assistant"):
                with st.spinner("🤖 智慧客服助理正在閱讀程式碼並分析中..."):
                    
                    # 建立完整的歷史對話文本，作為上下文 (Context)
                    history_text = "\n".join([
                        f'{m["role"]}: {m["content"]}' 
                        for m in st.session_state.messages
                    ])
    
                    # --- 程式碼自動提取區塊 (使用 inspect 和 strategy_object_map) ---
                    current_strategy_function = "N/A (找不到函數物件)"
                    
                    strategy_object = strategy_object_map.get(current_strategy)
                    
                    if strategy_object:
                        try:
                            current_strategy_function = inspect.getsource(strategy_object)
                        except (TypeError, OSError) as e:
                            current_strategy_function = f"N/A (獲取源碼失敗: {e})"
                    # -------------------------------------------------------------------
                    
                    # 建立包含程式碼和歷史紀錄的最終 Prompt
                    strategy_prompt = f"""
                    你是一位頂尖的量化金融策略解讀師，你的任務是為使用者提供極度聚焦於其問題的答案。
                                    
                    **[強制主題]：** **當前預設討論的策略是： {current_strategy}。**
                                    
                    **你的回答必須嚴格遵守以下規則：**
                    1. **翻譯指令 (核心規則)：** 你的解釋必須以「當前策略的 Python 函數程式碼」作為**唯一的邏輯依據**，但輸出結果必須將所有程式碼邏輯**完整翻譯**成**專業的金融或交易術語**。
                        * 範例：將 `data['Close'] > data['Recent_High']` 翻譯成「當收盤價高於近期高點時」。
                    2. **程式碼輸出禁令 (隱私與安全)：** 嚴禁在輸出結果中包含任何 Python 程式碼塊 (` ```python...``` `) 或任何程式碼片段。
                    3. **專注原則：** 你的所有回答都必須圍繞「{current_strategy}」進行解釋。
                    4. **對話延續：** 參考「歷史對話」理解上下文。
                    5. **格式：** 使用清晰的標題和點列式（如：進場條件、出場條件、使用的指標）來呈現邏輯，以提高非技術用戶的可讀性。
                    6. **【新增數據限制】**：**如果使用者詢問關於即時股票訊號、當前符合條件的股票名單、或今日/未來交易建議，必須禮貌且明確地拒絕。請解釋你的知識範圍僅限於「策略的規則程式碼」和「歷史回測結果」，無法存取即時股價數據或執行訊號篩選。**
                    
                    ---
                                    
                    **當前策略的 Python 函數程式碼 (內部參考知識庫)：**
                    {current_strategy_function}
                    
                    ---
                                    
                    **歷史對話紀錄：**
                    {history_text}
                                    
                    ---
                                    
                    **當前使用者問題：** {prompt}
                                    
                    請根據以上規則和程式碼內容，直接回答使用者問題。
                    """
                    
                    # 呼叫 LLM 服務
                    llm_response = llm_api_call(strategy_prompt)
                    
                    # 顯示 AI 的回答
                    st.markdown(llm_response)
    
            # 將 AI 的回答添加到聊天記錄
            st.session_state.messages.append({"role": "assistant", "content": llm_response})
            
            # 重新運行 Streamlit 以更新歷史紀錄和輸入框
            st.rerun()
    
        # =====================================================
        # LLM 應用結束
        # =====================================================

        
    # ----------------- 參數調整 -----------------
    with subtab_params:
        st.subheader("策略參數調整")
        
        # 這裡假設了 `strategies`, `page_prefix`, `strategy_params` 
        # 和 `st.session_state.backtest_params` 等變數在其他地方有定義
        filtered_strategies = [
            s for s in strategies
            if s[0] in st.session_state[f"{page_prefix}_selected_strategies"]
        ]
        
        if len(filtered_strategies) == 1:
            selected_strategy_name, _ = filtered_strategies[0]
            
            if selected_strategy_name in strategy_params:
                st.write(f"【{selected_strategy_name}】策略參數")
                
                params = {}
                for p, v in strategy_params[selected_strategy_name].items():
                    # 確保 number_input 的 key 唯一
                    unique_key = f"{selected_strategy_name}_{p}"
                    
                    if isinstance(v, int):
                        params[p] = st.number_input(p, value=v, key=unique_key)
                    elif isinstance(v, float):
                        params[p] = st.number_input(p, value=v, format="%.4f", key=unique_key)
                
                st.session_state.backtest_params = params
        else:
            st.info("請選擇單一策略以調整參數")
        
    # ================= 回測運算 (簡化進度條狀態) =================
    if st.session_state.backtest_is_running:
        
        # 定義 KEYS 和 LABELS (雖然 LABEL 不再用於狀態文字，但仍用於迴圈和結果儲存)
        DURATION_KEYS = ["3mo", "6mo", "ytd", "1y", "2y", "5y", "10y", "max"]
        DURATION_LABELS = ["3個月", "6個月", "今年以來", "一年", "二年", "五年", "十年", "全部資料"]
        
        # ------------------ 定義週期進度條 ------------------
        total_steps = len(DURATION_KEYS) 
        step = 0 
        
        # 在頂層的佔位符內定義進度條和狀態文本
        with status_container.container():
            progress_bar = st.progress(0)
            status_text = st.empty()
        # --------------------------------------------------------
        
        for p in DURATION_KEYS:
            
            # *** 修正點：簡化狀態文字到只有「正在回測 (X/8)...」 ***
            status_text.text(f"⏳ 正在回測 ({step + 1}/{total_steps})...") 
    
            # 抓資料：
            hist_data = load_hist_data([format_ticker(t) for t in tickers], p, None, None) 

            # 🔥 儲存當期的歷史資料
            st.session_state.backtest_hist_data[p] = hist_data

            
            # 檢查 hist_data 是否為空 (處理 NameError)
            if not hist_data:
                status_text.warning(f"⚠️ 資料抓取失敗，跳過此週期 ({step + 1}/{total_steps})...")
                
                step += 1
                progress_bar.progress(step / total_steps)
                
                continue 
    
            result_list = []
            trade_list = []
            
            for ticker, hist in hist_data.items():
                
                # 移除 stock_name 的計算，因為不再需要
                # stock_name = format_ticker_to_name(ticker, stock_dict)
                
                for strategy_name, strategy_func in filtered_strategies:
                    
                    df_signal = pd.DataFrame() 
                    
                    if len(filtered_strategies) == 1 and strategy_name in strategy_params:
                        df_signal = strategy_func(hist, **st.session_state.backtest_params)
                    else:
                        df_signal = strategy_func(hist)
                        
                    metrics, trades = backtest(df_signal, initial_capital, 0.000001425, 0.000003)
    
                    if metrics is None:
                        metrics = {
                            "錯誤": "訊號不足，該期間無交易。",
                            } 
                        
                    result_list.append({
                        "股票代碼": ticker,
                        "策略名稱": strategy_name,
                        **metrics
                    })

                    trades_to_append = []
                    if isinstance(trades, pd.DataFrame) and not trades.empty:
                        # 如果 backtest 回傳了 DataFrame (您的情況)，則將其轉換為 List of Dicts
                        trades_to_append = trades.to_dict('records') 
                    elif isinstance(trades, list) and len(trades) > 0:
                        # 如果 backtest 回傳了 List (備用格式)，則直接使用
                        trades_to_append = trades
                        
                    if isinstance(trades, list) and len(trades) > 0:
                        for t in trades:
                            t["股票代碼"] = ticker
                            t["策略名稱"] = strategy_name
                            trade_list.append(t)
                        
                    # *** 修正點：移除所有內層的狀態更新，保持進度條的簡潔 ***
                    pass
    
            
            # ---------------- 將股票代碼轉成股票名稱 + 移到第一欄 ----------------
            df_result = pd.DataFrame(result_list)
            df_result['股票'] = df_result['股票代碼'].apply(lambda x: format_ticker_to_name(x, stock_dict))
            df_result = df_result.drop(columns=["股票代碼"])
            cols = df_result.columns.tolist()
            cols.insert(0, cols.pop(cols.index("股票")))
            df_result = df_result[cols]
            
            df_result = df_result.drop(columns=["今天的訊號"], errors='ignore')
            
            st.session_state.backtest_results[p] = df_result
            st.session_state.backtest_trades[p] = pd.DataFrame(trade_list)
            
            # ---------------- 更新週期進度條 ----------------
            step += 1
            progress_bar.progress(step / total_steps)
            # --------------------------------------------------------
    
        # *** 確保進度條達到 100% 並顯示最終訊息 ***
        progress_bar.progress(1.0)
        status_text.text("✅ 所有回測週期完成！正在整理報告...") 
        time.sleep(1) 
        
        status_container.empty()
        
        st.session_state.backtest_is_running = False
        st.success("🎉 8 個期間回測完成！")
        st.rerun()
        
    # ================= 績效報告 =================
    with subtab_perf:

        if not st.session_state.backtest_is_running:
        
            DURATION_LABELS = ["3個月", "6個月", "今年以來", "一年", "二年", "五年", "十年", "全部資料"]
            DURATION_KEYS_DISPLAY = ["3mo", "6mo", "ytd", "1y", "2y", "5y", "10y", "max"]  
        
            period_tabs = st.tabs(DURATION_LABELS)
        
            for tab, key in zip(period_tabs, DURATION_KEYS_DISPLAY):
                with tab:
                    
                    # 1. 抓取該固定期間的資料
                    filtered_df = st.session_state.backtest_results.get(key, pd.DataFrame())
                    # 這裡的 trades_df 必須是回測運算區塊中已經根據 'key' 篩選好的結果
                    trades_df = st.session_state.backtest_trades.get(key, pd.DataFrame()) 
                    hist_data_period = st.session_state.backtest_hist_data.get(key, {})
        
                    current_label = DURATION_LABELS[DURATION_KEYS_DISPLAY.index(key)]
                    
                    # =====================================================
                    # 🚨 績效摘要顯示 (恢復到正確位置) 🚨
                    # =====================================================
                    if current_label == "全部歷史":
                        subtitle_text = "📈 全部歷史的策略績效"
                    elif current_label == "今年以來":
                        subtitle_text = "📈 今年以來的策略績效"
                    else:
                        subtitle_text = f"📈 近{current_label}的策略績效"

                    
                    st.subheader(subtitle_text)
                    column_cfg = {
                        col: st.column_config.TextColumn(width="medium")
                        for col in filtered_df.T.columns
                    }
                    st.dataframe(filtered_df.T, use_container_width=True,column_config=column_cfg)

                    # =====================================================
                    # ⭐ 報酬率曲線 (還原面積填充與顏色邏輯) ⭐
                    # =====================================================
                    st.markdown("---")
                    st.subheader("📈 報酬率曲線")
                    
                    import plotly.graph_objects as go
                    
                    if not trades_df.empty:
                        perf_df = trades_df.copy()
                        perf_df['淨利'] = pd.to_numeric(perf_df['淨利'], errors='coerce')
                        perf_df['Date'] = pd.to_datetime(perf_df['進場日期'], errors='coerce').dt.normalize()
                        perf_df = perf_df.dropna(subset=['Date', '淨利'])
                        
                        # 假設 initial_capital 已定義
                        initial_balance = initial_capital 
                        daily_profit = perf_df.groupby('Date')['淨利'].sum().reset_index()
                        daily_profit["累積淨利"] = daily_profit['淨利'].cumsum()
                        daily_profit["累積報酬率(%)"] = (daily_profit["累積淨利"] / initial_balance) * 100
                        
                        fig_perf = go.Figure()
                        x = daily_profit['Date'].tolist()
                        y = daily_profit['累積報酬率(%)'].tolist()
                        
                        # 重新實現 0 軸交叉和面積填充邏輯
                        for i in range(len(y)-1):
                            x_seg = [x[i], x[i+1]]
                            y_seg = [y[i], y[i+1]]
                            
                            # 情況 1 & 2: 不跨 0 軸
                            if y[i] >= 0 and y[i+1] >= 0:
                                color = 'red' # 都在 0 軸之上 (紅色)
                                fillcolor_rgba = 'rgba(255,0,0,0.3)'
                            elif y[i] <= 0 and y[i+1] <= 0:
                                color = 'green' # 都在 0 軸之下 (綠色)
                                fillcolor_rgba = 'rgba(0,128,0,0.3)'
                            else:
                                # 情況 3: 跨 0 軸 (需要分割線段)
                                
                                # 計算跨 0 的交叉點
                                delta_days = (x[i+1] - x[i]).days + (x[i+1] - x[i]).seconds / 86400
                                ratio = -y[i] / (y[i+1]-y[i])
                                cross_x = x[i] + pd.Timedelta(days=delta_days * ratio)
                                cross_y = 0
                                
                                # --- 前半段 ---
                                fig_perf.add_trace(go.Scatter(
                                    x=[x[i], cross_x],
                                    y=[y[i], cross_y],
                                    mode='lines',
                                    line=dict(color='red' if y[i]>0 else 'green', width=2),
                                    fill='tozeroy',
                                    fillcolor='rgba(255,0,0,0.3)' if y[i]>0 else 'rgba(0,128,0,0.3)',
                                    showlegend=False
                                ))
                                # --- 後半段 ---
                                fig_perf.add_trace(go.Scatter(
                                    x=[cross_x, x[i+1]],
                                    y=[cross_y, y[i+1]],
                                    mode='lines',
                                    line=dict(color='red' if y[i+1]>0 else 'green', width=2),
                                    fill='tozeroy',
                                    fillcolor='rgba(255,0,0,0.3)' if y[i+1]>0 else 'rgba(0,128,0,0.3)',
                                    showlegend=False
                                ))
                                continue # 跳過下面不跨軸的繪製
                    
                            # 不跨 0 軸的繪製
                            fig_perf.add_trace(go.Scatter(
                                x=x_seg,
                                y=y_seg,
                                mode='lines',
                                line=dict(color=color, width=2),
                                fill='tozeroy',
                                fillcolor=fillcolor_rgba,
                                showlegend=False
                            ))
                    
                        # 更新佈局
                        fig_perf.update_layout(
                            title=f"{current_label} | 累積報酬率",
                            yaxis_title="累積報酬率 (%)",
                            xaxis_title="日期",
                            hovermode='x unified'
                        )
                        fig_perf.update_yaxes(tickformat=".2f")
                        st.plotly_chart(fig_perf, use_container_width=True)
                    else:
                        st.info(f"⚠️ 期間 {current_label} 尚無交易紀錄。")
    
                    # =====================================================
                    # ⭐ 交易明細 (原 trade_tab 內容) ⭐
                    # =====================================================
                    st.markdown("---")
                    st.subheader("📋 交易明細")

                    # 在呈現前，移除不想顯示的欄位
                    hide_cols = ['手續費', '稅費', '報酬率.', '股票代碼', '策略名稱']
                    
                    trades_display = trades_df.drop(columns=[c for c in hide_cols if c in trades_df.columns])
                    
                    if not trades_df.empty:
                        st.dataframe(trades_display, use_container_width=True)
                    else:
                        st.info(f"⚠️ 期間 {current_label} 無交易資料。")
                        
                    # =====================================================
                    # ⭐ LLM 應用：績效智慧摘要與風險剖析 (絕對在最下方) ⭐
                    # =====================================================
                    st.markdown("---") 
                    
                    if not filtered_df.empty:
                        st.subheader("📊 績效智慧摘要與風險剖析")
    
                        metrics_summary = filtered_df.T.to_markdown(index=True)
                        current_ticker = st.session_state.get('backtest_ticker', '未知股票')
                        
                        # 使用上一個回覆中修正後的 perf_prompt 內容
                        perf_prompt = f"""
                        你是一位經驗豐富的金融風險分析師，你的任務是解讀量化回測結果，並為用戶提供專業的中文摘要報告與**風險剖析**。
                        你的目標是**客觀地報告**數據的事實，而不是指導用戶操作。
                        
                        **[最終且嚴格的安全合規規則]：**
                        1. **輸出限制：** 你的報告必須是基於歷史數據的客觀事實陳述。
                        2. **語氣限制 (絕對禁止)：** **嚴禁**使用任何形式的「操作建議」、「應當」、「避免」、「推薦」、「請勿」等規範性（引導用戶行動）的詞彙。
                        3. **角色定位：** 你是一名數據分析師，不是投資顧問。你的結論只能是「此策略的歷史表現特性」，絕不能是「用戶應採取的行動」。
                        
                        **分析數據：**
                        * 股票代碼: {current_ticker}
                        * 策略名稱: {current_strategy}
                        * 回測期間: {current_label}
                        
                        嚴格根據績效指標數據 (Markdown Table 格式):
                        {metrics_summary}
                        
                        **分析重點要求 (請進行數值推理與客觀剖析)：**
                        1. 簡述**累積報酬率**和**年化報酬率**的歷史表現。
                        2. 評估 **Sharpe Ratio (夏普比率)**，客觀描述其**風險調整後收益**特性。
                        3. **風險剖析 (Risk Analysis)：** 強調 **Max Drawdown (最大資金回撤)**，並指出這代表的**歷史最大風險暴露**，說明此數據對該策略類型的意義。
                        4. **客觀總結**：你的最後一句必須是**嚴格的客觀總結**，總結該策略在該期間的**歷史表現特性**和**潛在的適用市場環境**。
                        
                        """
                        
                        # 實際呼叫 LLM API
                        with st.spinner("📊 AI 分析師正在解讀回測數據並生成報告..."):
                            llm_report = llm_api_call(perf_prompt)
                            
                        # 顯示 LLM 的結果
                        st.success(llm_report)
