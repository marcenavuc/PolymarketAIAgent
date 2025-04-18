import os
from datetime import datetime, timedelta

import pandas as pd
import requests

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
