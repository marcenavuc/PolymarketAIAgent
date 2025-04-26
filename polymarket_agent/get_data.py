import os
from datetime import datetime, timedelta

import pandas as pd
import requests

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

DT_FORMAT = '%Y-%m-%d'


def add_days(dt: str, days: int, dt_format: str = DT_FORMAT) -> str:
    return datetime.strftime(datetime.strptime(dt, dt_format) + timedelta(days=days), dt_format)


def get_klines(symbol: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
    sdt = int(datetime.strptime(start_date, DT_FORMAT).timestamp() * 1000)
    edt = int(datetime.strptime(end_date, DT_FORMAT).timestamp() * 1000)

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": sdt,
        "endTime": edt
    }

    klines = requests.get('https://api.binance.com/api/v3/klines', params=params)
    klines_df = pd.DataFrame(klines.json())

    klines_df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                         'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    klines_df['open_time'] = pd.to_datetime(klines_df['open_time'], unit='ms')
    klines_df['close_time'] = pd.to_datetime(klines_df['close_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        klines_df[col] = klines_df[col].apply(lambda x: float(x))

    klines_df = klines_df[['open_time', 'close_time', 'volume', 'open', 'close']]
    klines_df['diff'] = klines_df['close'] - klines_df['open']

    return klines_df


def get_crypto_news_from_cryptopanic(
        keyword: str,
        start_date: str,
        end_date: str,
        kind: str = 'news',
        filter: str = 'important',
) -> pd.DataFrame:
    url = 'https://cryptopanic.com/api/v1/posts/'
    params = {
        'auth_token': os.environ["CRYPTOPANIC_API_KEY"],
        'currencies': keyword,
        'kind': kind,
        'filter': filter
    }

    data = requests.get(url, params=params).json()
    result = data['results']

    for i in range(data['count']):
        url = data['next']

        if url is None:
            break

        data = requests.get(url).json()
        result.extend(data['results'])

    news_df = pd.DataFrame(result)
    for col in ['positive', 'important', 'liked']:
        news_df[col] = news_df['votes'].apply(lambda x: x[col])

    news_df = news_df[['published_at', 'title', 'positive', 'important', 'liked']]
    news_df['published_at'] = pd.to_datetime(news_df['published_at']).dt.tz_convert(None)

    news_df['date'] = news_df['published_at'].apply(lambda x: str(x)[:10])
    news_df = news_df.drop(columns=['published_at'])
    news_df = news_df[(start_date <= news_df['date']) & (news_df['date'] <= end_date)]

    return news_df.sort_values(['date', 'important'], ascending=False)


def get_data(tiket, start_date, end_date):
    news_df = get_crypto_news_from_cryptopanic(tiket, start_date, end_date)
    klines_df = get_klines(tiket + 'USDT', start_date, end_date)

    return news_df, klines_df


def get_klines_ta(symbol: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
    sdt = int(datetime.strptime(start_date, DT_FORMAT).timestamp() * 1000)
    edt = int(datetime.strptime(end_date, DT_FORMAT).timestamp() * 1000)

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": sdt,
        "endTime": edt
    }

    klines = requests.get('https://api.binance.com/api/v3/klines', params=params)
    klines_df = pd.DataFrame(klines.json())

    klines_df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                         'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    klines_df['open_time'] = pd.to_datetime(klines_df['open_time'], unit='ms')
    klines_df['close_time'] = pd.to_datetime(klines_df['close_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        klines_df[col] = klines_df[col].apply(lambda x: float(x))

    klines_df['date'] = klines_df['open_time']
    # klines_df = klines_df[['date', 'close_time', 'volume', 'open', 'close']]
    klines_df['diff'] = klines_df['close'] - klines_df['open']
    klines_df['taker_buy_base_asset_volume'] = klines_df['taker_buy_base_asset_volume'].astype('float')
    
    # Процентное изменение цены
    klines_df['pct_change'] = klines_df['close'].pct_change() * 100  
    
    # Среднеисторическая волатильность (скользящее стандартное отклонение)
    klines_df['volatility_7d'] = klines_df['pct_change'].rolling(7).std()  
    
    # Скользящие средние (MA)
    klines_df['MA_7'] = klines_df['close'].rolling(7).mean()  
    klines_df['MA_21'] = klines_df['close'].rolling(21).mean()  
    
    # Разница между ценой и MA (отклонение от тренда)
    klines_df['price_ma7_diff'] = klines_df['close'] - klines_df['MA_7']

    # RSI (Relative Strength Index) - Индекс относительной силы, RSI > 70 → цена может упасть, RSI < 30 → цена может вырасти.
    rsi = RSIIndicator(klines_df['close'], window=14)
    klines_df['RSI'] = rsi.rsi()
    
    # MACD (Moving Average Convergence Divergence) – тренд и моменты разворота.
    macd = MACD(klines_df['close'])
    klines_df['MACD'] = macd.macd()
    klines_df['MACD_signal'] = macd.macd_signal()
    
    # Bollinger Bands – волатильность и границы диапазона.
    bb = BollingerBands(klines_df['close'])
    klines_df['BB_upper'] = bb.bollinger_hband()
    klines_df['BB_lower'] = bb.bollinger_lband()
    
    # Средний объем за N дней
    klines_df['volume_ma7'] = klines_df['volume'].rolling(7).mean()  
    
    # Объемный профиль (отношение buy/sell объема)
    klines_df['buy_volume_ratio'] = klines_df['taker_buy_base_asset_volume'] / klines_df['volume']  
    
    # Аномалии объема (Z-score)
    klines_df['volume_zscore'] = (klines_df['volume'] - klines_df['volume'].mean()) / klines_df['volume'].std()  
    
    klines_df['day_of_week'] = klines_df['date'].dt.dayofweek  
    
    # Бычье/медвежье поглощение
    klines_df['bullish_engulfing'] = (
        (klines_df['close'] > klines_df['open']) & 
        (klines_df['close'].shift(1) < klines_df['open'].shift(1)) & 
        (klines_df['close'] > klines_df['open'].shift(1)) & 
        (klines_df['open'] < klines_df['close'].shift(1))
    ).astype(int)  
    
    # Доджи (нерешительность рынка)
    klines_df['doji'] = (abs(klines_df['open'] - klines_df['close']) / (klines_df['high'] - klines_df['low']) < 0.1).astype(int) 
    
    # Лаговые значения (для предсказания)
    klines_df['close_lag1'] = klines_df['close'].shift(1)  
    klines_df['close_lag3'] = klines_df['close'].shift(3)  
    
    # Целевая переменная (например, цена через N дней)
    klines_df['target_close_3d'] = klines_df['close'].shift(-3)  
    return klines_df


def get_data_ta(tiket, start_date, end_date):
    news_df = get_crypto_news_from_cryptopanic(tiket, start_date, end_date)
    klines_df = get_klines_ta(tiket + 'USDT', start_date, end_date)

    return news_df, klines_df
