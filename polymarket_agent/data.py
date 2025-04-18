import os

import pandas as pd
import requests


def get_klines(symbol: str, interval: str = "1d", limit: int = 7) -> pd.DataFrame:
    klines = requests.get(f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}')

    klines_df = pd.DataFrame(klines.json())

    klines_df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                         'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    klines_df['open_time'] = pd.to_datetime(klines_df['open_time'], unit='ms')
    klines_df['close_time'] = pd.to_datetime(klines_df['close_time'], unit='ms')
    klines_df = klines_df[['open_time', 'close_time', 'volume', 'low', 'close']]

    return klines_df


def get_crypto_news_from_cryptopanic(
        keyword: str,
        kind: str = 'news',
        filter: str = 'important',
        limit: int = 7
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
    news_df['positive'] = news_df['votes'].apply(lambda x: x['positive'])
    news_df['important'] = news_df['votes'].apply(lambda x: x['important'])
    news_df['liked'] = news_df['votes'].apply(lambda x: x['liked'])

    news_df = news_df[['published_at', 'title', 'positive', 'important', 'liked']]
    news_df.sort_values(['published_at', 'important'], ascending=False)

    return news_df
