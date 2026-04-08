"""
v4 — 하이브리드(분류+회귀) + 캘리브레이션 + 대안 피처 + NLP 감성 (최종)

변경점 (v3 대비):
  - 모델: 회귀 앙상블 → 분류 앙상블(방향) + 회귀 앙상블(크기) 하이브리드
  - 데이터: 2년 → 5년 (462건 → 1,200건)
  - 피처: 22개 → 33개 (대안 6개 + NLP 감성 5개 추가)
  - 확률: Isotonic Regression 캘리브레이션 + 동적 시그널
  - NLP: 네이버 뉴스 크롤링 + KR-FinBert + 2단계 필터링
  - 피처 선택: 중요도 기반 자동 선택
  - 튜닝: Optuna 분류/회귀 각각 독립 탐색
"""
import xgboost as xgb
import lightgbm as lgbm
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import optuna
import os
import time
import json
import re
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from transformers import pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [v4 추가] KR-FinBert 감성 분석기 (싱글톤)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_sentiment_pipeline = None

def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        print("  [NLP] KR-FinBert 모델 로드 중... (최초 1회)")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="snunlp/KR-FinBert-SC",
            tokenizer="snunlp/KR-FinBert-SC",
            top_k=None,
        )
        print("  [NLP] 모델 로드 완료")
    return _sentiment_pipeline


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [v4 추가] 네이버 뉴스 크롤링 + 캐싱
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "news_cache")

headers_naver = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def _get_cache_path(stock_name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name = re.sub(r'[^\w]', '_', stock_name)
    return os.path.join(CACHE_DIR, f"news_{safe_name}.json")


def _load_cache(stock_name):
    path = _get_cache_path(stock_name)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _save_cache(stock_name, cache_data):
    path = _get_cache_path(stock_name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)


def crawl_naver_news_week(query, start_date, end_date):
    ds = start_date.replace("-", ".")
    de = end_date.replace("-", ".")
    nso_start = start_date.replace("-", "")
    nso_end = end_date.replace("-", "")

    url = "https://search.naver.com/search.naver"
    params = {
        "where": "news", "query": query, "sort": 1,
        "ds": ds, "de": de,
        "nso": f"so:dd,p:from{nso_start}to{nso_end}",
        "start": 1,
    }

    titles = []
    try:
        resp = requests.get(url, params=params, headers=headers_naver, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')

            news_blocks = soup.select('.group_news .sds-comps-full-layout')
            if news_blocks:
                for block in news_blocks:
                    links = block.select('a')
                    if links:
                        title = links[0].get_text(strip=True)
                        if title and 10 < len(title) < 200:
                            titles.append(title)

            if not titles:
                for sel in ['a.news_tit', '.news_tit']:
                    items = soup.select(sel)
                    if items:
                        for item in items[:15]:
                            title = item.get_text(strip=True)
                            if title and len(title) > 5:
                                titles.append(title)
                        break

            if not titles:
                for a in soup.select('.group_news a'):
                    text = a.get_text(strip=True)
                    href = a.get('href', '')
                    if (text and 10 < len(text) < 200
                            and any(d in href for d in ['news', 'article', 'munhwa', 'newsis'])):
                        if text not in titles:
                            titles.append(text)

            titles = list(dict.fromkeys(titles))[:15]
    except Exception:
        pass

    return titles


def crawl_news_for_period(stock_name, start_date, end_date):
    cache = _load_cache(stock_name)
    new_data = False

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    current = start
    crawled_count = 0
    cached_count = 0

    while current < end:
        week_end = min(current + timedelta(days=6), end)
        week_key = current.strftime("%Y-%m-%d")

        if week_key not in cache:
            titles = crawl_naver_news_week(
                stock_name, current.strftime("%Y-%m-%d"), week_end.strftime("%Y-%m-%d"),
            )
            cache[week_key] = titles
            new_data = True
            crawled_count += 1
            time.sleep(np.random.uniform(0.5, 1.0))
            if crawled_count % 50 == 0:
                print(f"    크롤링 진행: {crawled_count}주 완료...")
        else:
            cached_count += 1

        current = week_end + timedelta(days=1)

    if new_data:
        _save_cache(stock_name, cache)

    print(f"  [크롤링] 신규: {crawled_count}주 | 캐시: {cached_count}주 | "
          f"총 뉴스: {sum(len(v) for v in cache.values())}건")
    return cache


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [v4 추가] 감성 분석 + 2단계 필터링 + 피처 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def analyze_sentiment_batch(titles, batch_size=32):
    if not titles:
        return []
    pipe = get_sentiment_pipeline()
    results = []
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i + batch_size]
        try:
            batch_results = pipe(batch, truncation=True, max_length=128)
            for item in batch_results:
                scores = {label_info['label'].lower(): label_info['score'] for label_info in item}
                results.append(scores)
        except Exception:
            for _ in batch:
                results.append({'positive': 0.33, 'negative': 0.33, 'neutral': 0.34})
    return results


def build_sentiment_features(stock_name, date_index, start_date, end_date):
    print(f"\n  [감성 분석] 뉴스 크롤링 시작...")
    news_cache = crawl_news_for_period(stock_name, start_date, end_date)

    print(f"  [감성 분석] 1차 필터 (키워드 관련성)...")
    filtered_titles = []
    filtered_week_keys = []
    total_raw = 0

    for week_key in sorted(news_cache.keys()):
        titles = news_cache[week_key]
        for title in titles:
            total_raw += 1
            if is_relevant_news(title, stock_name):
                filtered_titles.append(title)
                filtered_week_keys.append(week_key)

    print(f"  [1차 필터] {total_raw}건 → {len(filtered_titles)}건 "
          f"(관련 기사 {len(filtered_titles)/max(total_raw,1):.0%})")

    if not filtered_titles:
        print(f"  관련 기사 0건 — 감성 피처 0으로 채움")
        sentiment_df = pd.DataFrame(index=date_index)
        for col in SENTIMENT_FEATURES:
            sentiment_df[col] = 0
        return sentiment_df

    print(f"  [감성 분석] KR-FinBert 분석 중...")
    all_scores = analyze_sentiment_batch(filtered_titles)

    print(f"  [감성 분석] 2차 필터 (감성 강도 > 0.3)...")
    week_scores = {}
    strong_count = 0

    for week_key, title, scores in zip(filtered_week_keys, filtered_titles, all_scores):
        # 감성 강도 필터 (약한 감성 제거)
        sentiment = scores.get('positive', 0) - scores.get('negative', 0)
        if abs(sentiment) < 0.3:
            continue
        strong_count += 1
        if week_key not in week_scores:
            week_scores[week_key] = []
        week_scores[week_key].append({
            'sentiment': sentiment,
            'is_positive': 1 if sentiment > 0.1 else 0,
        })

    print(f"  [2차 필터] {len(filtered_titles)}건 → {strong_count}건 "
          f"(강한 감성 {strong_count/max(len(filtered_titles),1):.0%})")

    weekly_sentiment = {}
    for week_key, scores_list in week_scores.items():
        sentiments = [s['sentiment'] for s in scores_list]
        positives = [s['is_positive'] for s in scores_list]
        weekly_sentiment[week_key] = {
            'avg_sentiment': np.mean(sentiments),
            'positive_ratio': np.mean(positives),
            'count': len(scores_list),
            'std': np.std(sentiments) if len(sentiments) > 1 else 0,
        }

    print(f"  [감성 분석] 최종: {strong_count}건 사용 "
          f"({strong_count/max(total_raw,1):.0%} of 원본)")

    sentiment_df = pd.DataFrame(index=date_index)
    sentiment_df['News_Sentiment'] = np.nan
    sentiment_df['News_Positive_Ratio'] = np.nan
    sentiment_df['News_Count'] = 0.0
    sentiment_df['News_Sentiment_Std'] = np.nan

    for week_key, data in weekly_sentiment.items():
        week_date = pd.Timestamp(week_key)
        week_end = week_date + timedelta(days=6)
        mask = (sentiment_df.index >= week_date) & (sentiment_df.index <= week_end)
        sentiment_df.loc[mask, 'News_Sentiment'] = data['avg_sentiment']
        sentiment_df.loc[mask, 'News_Positive_Ratio'] = data['positive_ratio']
        sentiment_df.loc[mask, 'News_Count'] = data['count']
        sentiment_df.loc[mask, 'News_Sentiment_Std'] = data['std']

    sentiment_df = sentiment_df.ffill().fillna(0)
    sentiment_df['News_Momentum'] = (
        sentiment_df['News_Sentiment'].rolling(5).mean()
        - sentiment_df['News_Sentiment'].rolling(20).mean()
    ).fillna(0)

    return sentiment_df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 종목 검색
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
headers_yf = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def search(company_name):
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={company_name}"
    response = requests.get(url, headers=headers_yf, timeout=5)
    results = response.json().get("quotes", [])
    return results[0]["symbol"] if results else None

def translate_to_english(text):
    return GoogleTranslator(source='ko', target='en').translate(text)

def find_company_symbol(name):
    for word in name.split():
        if not word.isascii():
            word = translate_to_english(word)
        symbol = search(word)
        if symbol:
            return symbol
    return None

def get_korean_name(user_input):
    for word in user_input.split():
        if not word.isascii():
            return word
    return user_input


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [v4 추가] 섹터 매핑
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def guess_sector_etf(symbol):
    symbol_upper = symbol.upper()
    semi_keywords = ['005930', '000660', '042700']
    tech_keywords = ['035420', '035720', '263750']
    finance_keywords = ['105560', '055550', '086790']
    bio_keywords = ['207940', '068270', '035900']

    for kw in semi_keywords:
        if kw in symbol_upper:
            return 'SOXX'
    for kw in tech_keywords:
        if kw in symbol_upper:
            return 'XLK'
    for kw in finance_keywords:
        if kw in symbol_upper:
            return 'XLF'
    for kw in bio_keywords:
        if kw in symbol_upper:
            return 'XLV'

    try:
        info = yf.Ticker(symbol).info
        sector = info.get('sector', '')
        if 'Technol' in sector: return 'XLK'
        elif 'Financ' in sector: return 'XLF'
        elif 'Health' in sector: return 'XLV'
        elif 'Energy' in sector: return 'XLE'
        elif 'Consumer' in sector: return 'XLY'
    except Exception:
        pass
    return 'SPY'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 피처 정의 (v3: 22개 → v4: 33개)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TECH_FEATURES = [
    'MA20', 'RSI_14', 'MACD', 'Bollinger_%B', 'Vol_MA5', 'Pct_change',
    'ATR_14', 'Stochastic_%K', 'Stochastic_%D', 'OBV_change',
    'MACD_signal', 'MACD_hist', 'Lag_1', 'Lag_2', 'Lag_3',
]
MACRO_FEATURES = ['VIX', 'VIX_change', 'DXY_change', 'US10Y_change', 'SP500_change']
# [v4 추가]
ALTERNATIVE_FEATURES = [
    'Sector_RelStr_5', 'Sector_RelStr_20',
    'Vol_Surprise', 'Spread_HighLow_20', 'Momentum_Cross', 'Gap_Return',
]
SENTIMENT_FEATURES = [
    'News_Sentiment', 'News_Positive_Ratio', 'News_Count',
    'News_Sentiment_Std', 'News_Momentum',
]
CALENDAR_FEATURES = ['DayOfWeek', 'Month']

RELEVANT_KEYWORDS = [
    '실적', '매출', '영업이익', '순이익', '적자', '흑자', '어닝',
    '분기', '연간', '전망', '가이던스', '컨센서스',
    '주가', '목표가', '상향', '하향', '투자의견', '매수', '매도',
    '신고가', '신저가', '급등', '급락', '반등', '하락',
    '외국인', '기관', '순매수', '순매도', '공매도', '대차',
    '수주', '계약', '투자', '인수', '합병', '신사업', '양산', '점유율', '출하', '가동률',
    '배당', '자사주', '주주환원',
    '반도체', 'HBM', 'AI', '파운드리', 'DRAM', 'NAND', '메모리',
]

def is_relevant_news(title, stock_name):
    if stock_name not in title:
        return False
    return any(kw in title for kw in RELEVANT_KEYWORDS)

ALL_FEATURES = (TECH_FEATURES + MACRO_FEATURES + ALTERNATIVE_FEATURES
                + SENTIMENT_FEATURES + CALENDAR_FEATURES)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 피처 계산
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def calculate_features(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df["MA20"] = df['Close'].rolling(window=20).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss))

    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26

    ma20_bb = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    upper = ma20_bb + (std20 * 2)
    lower = ma20_bb - (std20 * 2)
    df['Bollinger_%B'] = (df['Close'] - lower) / (upper - lower)

    df["Vol_MA5"] = df['Volume'].rolling(window=5).mean()
    df["Pct_change"] = df['Close'].pct_change()

    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = true_range.rolling(14).mean()

    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['Stochastic_%K'] = (df['Close'] - low_14) / (high_14 - low_14) * 100
    df['Stochastic_%D'] = df['Stochastic_%K'].rolling(3).mean()

    obv = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    df['OBV_change'] = obv.pct_change(5)

    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    df['Lag_1'] = df['Pct_change'].shift(1)
    df['Lag_2'] = df['Pct_change'].shift(2)
    df['Lag_3'] = df['Pct_change'].shift(3)

    # [v4 추가] 대안 피처
    vol_ma20 = df['Volume'].rolling(20).mean()
    df['Vol_Surprise'] = df['Volume'] / vol_ma20

    df['Spread_HighLow_20'] = (
        df['High'].rolling(20).max() - df['Low'].rolling(20).min()
    ) / df['Close']

    ma5 = df['Close'].rolling(5).mean()
    ma20 = df['Close'].rolling(20).mean()
    df['Momentum_Cross'] = (ma5 - ma20) / ma20

    df['Gap_Return'] = df['Open'] / df['Close'].shift(1) - 1

    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month

    return df


def add_sector_relative_strength(df, symbol, start_date, end_date):
    sector_etf = guess_sector_etf(symbol)
    print(f"  섹터 ETF: {sector_etf}")
    try:
        etf_data = yf.download(sector_etf, start=start_date, end=end_date, progress=False)
        if isinstance(etf_data.columns, pd.MultiIndex):
            etf_data.columns = etf_data.columns.droplevel(1)
        etf_ret_5 = etf_data['Close'].pct_change(5)
        etf_ret_20 = etf_data['Close'].pct_change(20)
        stock_ret_5 = df['Close'].pct_change(5)
        stock_ret_20 = df['Close'].pct_change(20)
        df['Sector_RelStr_5'] = stock_ret_5 - etf_ret_5.reindex(df.index).ffill()
        df['Sector_RelStr_20'] = stock_ret_20 - etf_ret_20.reindex(df.index).ffill()
    except Exception as e:
        print(f"  섹터 ETF 실패: {e}")
        df['Sector_RelStr_5'] = 0
        df['Sector_RelStr_20'] = 0
    return df


def fetch_macro_data(start_date, end_date):
    tickers = {
        'VIX': '^VIX', 'DXY': 'DX-Y.NYB',
        'US10Y': '^TNX', 'SP500': '^GSPC',
    }
    macro_df = pd.DataFrame()
    for name, ticker in tickers.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            macro_df[name] = data['Close']
        except Exception:
            pass
    return macro_df


def add_macro_features(df, macro_df):
    if 'VIX' in macro_df.columns:
        df = df.join(macro_df[['VIX']], how='left')
        df['VIX'] = df['VIX'].ffill()
        df['VIX_change'] = df['VIX'].pct_change(5)
    else:
        df['VIX'] = 0
        df['VIX_change'] = 0

    for name in ['DXY', 'US10Y', 'SP500']:
        col_name = f"{name}_change"
        if name in macro_df.columns:
            df = df.join(macro_df[[name]].rename(columns={name: col_name + '_raw'}), how='left')
            df[col_name + '_raw'] = df[col_name + '_raw'].ffill()
            df[col_name] = df[col_name + '_raw'].pct_change(5)
            df.drop(columns=[col_name + '_raw'], inplace=True)
        else:
            df[col_name] = 0
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터 준비
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def stock_data(symbol, stock_name, years=5):
    today = datetime.today()
    start = today - relativedelta(years=years)
    start_str = start.strftime("%Y-%m-%d")
    end_str = today.strftime("%Y-%m-%d")

    print(f"  데이터 기간: {start_str} ~ {end_str} ({years}년)")

    df = yf.download(symbol, start=start_str, end=end_str)
    df = calculate_features(df)
    df = add_sector_relative_strength(df, symbol, start_str, end_str)

    macro_df = fetch_macro_data(start_str, end_str)
    df = add_macro_features(df, macro_df)

    # NLP 감성 피처 (최근 2년만)
    news_start = (today - relativedelta(years=2)).strftime("%Y-%m-%d")
    sentiment_df = build_sentiment_features(stock_name, df.index, news_start, end_str)
    for col in SENTIMENT_FEATURES:
        if col in sentiment_df.columns:
            df[col] = sentiment_df[col].reindex(df.index).ffill().fillna(0)
        else:
            df[col] = 0

    # 타겟
    df["Target_Return"] = df["Close"].pct_change().shift(-1)
    df["Target_Direction"] = (df["Target_Return"] > 0).astype(int)

    df = df[ALL_FEATURES + ['Target_Return', 'Target_Direction']]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    print(f"  최종 샘플 수: {len(df)}건")
    return df


def compare_data(symbol, stock_name, feature_list):
    df_pred = yf.download(symbol, period="60d", progress=False)
    df_pred = calculate_features(df_pred)
    if isinstance(df_pred.columns, pd.MultiIndex):
        df_pred.columns = df_pred.columns.droplevel(1)

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - relativedelta(days=90)).strftime("%Y-%m-%d")

    df_pred = add_sector_relative_strength(df_pred, symbol, start_date, end_date)
    macro_df = fetch_macro_data(start_date, end_date)
    df_pred = add_macro_features(df_pred, macro_df)

    sentiment_df = build_sentiment_features(stock_name, df_pred.index, start_date, end_date)
    for col in SENTIMENT_FEATURES:
        if col in sentiment_df.columns:
            df_pred[col] = sentiment_df[col].reindex(df_pred.index).ffill().fillna(0)
        else:
            df_pred[col] = 0

    df_pred.replace([np.inf, -np.inf], np.nan, inplace=True)
    last_close = float(df_pred['Close'].iloc[-1])
    features_row = df_pred[feature_list].iloc[-1:]

    return features_row, last_close


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [v4 추가] 피처 선택
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def select_features_by_importance(X_train, y_train, X_test, y_test, threshold=0.015):
    temp_model = xgb.XGBClassifier(
        max_depth=3, learning_rate=0.01, n_estimators=300,
        subsample=0.6, colsample_bytree=0.6,
        objective='binary:logistic', verbosity=0, eval_metric='logloss',
    )
    temp_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    importance = pd.Series(
        temp_model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    selected = importance[importance >= threshold].index.tolist()
    removed = [f for f in X_train.columns if f not in selected]

    # 감성 피처 최소 1개 보존
    sentiment_in_selected = [f for f in selected if f in SENTIMENT_FEATURES]
    if not sentiment_in_selected:
        best_sentiment = None
        best_imp = -1
        for f in SENTIMENT_FEATURES:
            if f in importance.index and importance[f] > best_imp:
                best_imp = importance[f]
                best_sentiment = f
        if best_sentiment:
            selected.append(best_sentiment)
            if best_sentiment in removed:
                removed.remove(best_sentiment)

    print(f"  피처 선택: {len(X_train.columns)}개 → {len(selected)}개")
    print(f"  제거됨: {removed}")
    print(f"  Top 10:")
    for feat, imp in importance.head(10).items():
        marker = " *" if feat in SENTIMENT_FEATURES else ""
        print(f"    {feat:>25s}: {imp:.4f}{marker}")

    return selected


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [v4 변경] Optuna — 분류/회귀 분리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def optimize_classifier_params(X, y, n_splits=3, n_trials=50):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500, step=100),
            'gamma': trial.suggest_float('gamma', 0.5, 3.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 30),
            'subsample': trial.suggest_float('subsample', 0.4, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.8),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.7, 1.5),
            'objective': 'binary:logistic', 'eval_metric': 'logloss', 'verbosity': 0,
        }
        da_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            da_scores.append(accuracy_score(y_te, model.predict(X_te)))
        return np.mean(da_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n  [Optuna 분류] 최적 정확도: {study.best_value:.2%}")
    return study.best_params


def optimize_regressor_params(X, y, n_splits=3, n_trials=30):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 300, 1200, step=100),
            'gamma': trial.suggest_float('gamma', 0.5, 3.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 30),
            'subsample': trial.suggest_float('subsample', 0.4, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.8),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
            'objective': 'reg:squarederror', 'verbosity': 0,
        }
        mae_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            mae_scores.append(mean_absolute_error(y_te, model.predict(X_te)))
        return np.mean(mae_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n  [Optuna 회귀] 최적 MAE: {study.best_value:.6f}")
    return study.best_params


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [v4 추가] 확률 캘리브레이션
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ProbabilityCalibrator:
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False
        self.confidence_bins = None

    def fit(self, raw_probs, y_true):
        if len(raw_probs) < 20:
            print("  캘리브레이션 샘플 부족 — 스킵")
            return
        self.calibrator.fit(raw_probs, y_true)
        self.is_fitted = True
        self._build_confidence_table(raw_probs, y_true)

    def _build_confidence_table(self, raw_probs, y_true):
        calibrated = self.calibrator.predict(raw_probs)
        confidence = np.abs(calibrated - 0.5) * 2
        directions = (calibrated > 0.5).astype(int)
        correct = (directions == y_true).astype(int)

        bins = [0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
        labels = ['0-5%', '5-10%', '10-20%', '20-30%', '30-50%', '50%+']
        bin_idx = np.clip(np.digitize(confidence, bins) - 1, 0, len(labels) - 1)

        self.confidence_bins = {}
        print("\n  확신도 구간별 적중률:")
        print(f"  {'구간':>10s} | {'적중률':>8s} | {'샘플수':>6s}")
        print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*6}")

        for i, label in enumerate(labels):
            mask = bin_idx == i
            if mask.sum() > 0:
                acc = correct[mask].mean()
                count = mask.sum()
                self.confidence_bins[label] = {
                    'accuracy': round(acc, 4), 'count': int(count),
                    'min_conf': bins[i], 'max_conf': bins[i + 1],
                }
                print(f"  {label:>10s} | {acc:>7.2%} | {count:>6d}")
            else:
                self.confidence_bins[label] = {
                    'accuracy': 0, 'count': 0,
                    'min_conf': bins[i], 'max_conf': bins[i + 1],
                }

    def calibrate(self, raw_prob):
        if not self.is_fitted:
            return raw_prob
        return float(self.calibrator.predict(np.array([raw_prob]))[0])

    def get_dynamic_signal(self, calibrated_prob):
        confidence = abs(calibrated_prob - 0.5) * 2
        direction = "상승" if calibrated_prob > 0.5 else "하락"

        if not self.confidence_bins:
            if confidence >= 0.3:
                signal = "강한 매수" if direction == "상승" else "강한 매도"
            elif confidence >= 0.1:
                signal = "약한 매수" if direction == "상승" else "약한 매도"
            else:
                signal = "관망"
            return signal, confidence, None

        current_bin = None
        for label, info in self.confidence_bins.items():
            if info['min_conf'] <= confidence < info['max_conf']:
                current_bin = info
                break

        if current_bin is None or current_bin['count'] < 5:
            return "관망 (데이터 부족)", confidence, current_bin

        actual_acc = current_bin['accuracy']
        if actual_acc >= 0.60:
            signal = f"강한 매수 (적중 {actual_acc:.0%})" if direction == "상승" \
                else f"강한 매도 (적중 {actual_acc:.0%})"
        elif actual_acc >= 0.55:
            signal = f"약한 매수 (적중 {actual_acc:.0%})" if direction == "상승" \
                else f"약한 매도 (적중 {actual_acc:.0%})"
        else:
            signal = f"관망 (적중 {actual_acc:.0%})"

        return signal, confidence, current_bin


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [v4 변경] 하이브리드 앙상블 모델
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class HybridEnsembleModel:
    def __init__(self, clf_params, reg_params):
        self.clf_params = clf_params
        self.reg_params = reg_params
        self.xgb_clf = self.lgbm_clf = self.xgb_reg = self.lgbm_reg = None
        self.clf_xgb_weight = self.clf_lgbm_weight = 0.5
        self.reg_xgb_weight = self.reg_lgbm_weight = 0.5

    def _make_lgbm_clf_params(self):
        return {
            'max_depth': self.clf_params.get('max_depth', 3),
            'learning_rate': self.clf_params.get('learning_rate', 0.01),
            'n_estimators': self.clf_params.get('n_estimators', 500),
            'subsample': self.clf_params.get('subsample', 0.6),
            'colsample_bytree': self.clf_params.get('colsample_bytree', 0.6),
            'reg_alpha': self.clf_params.get('reg_alpha', 0.1),
            'reg_lambda': self.clf_params.get('reg_lambda', 1.5),
            'min_child_weight': self.clf_params.get('min_child_weight', 10),
            'objective': 'binary', 'verbosity': -1,
        }

    def _make_lgbm_reg_params(self):
        return {
            'max_depth': self.reg_params.get('max_depth', 3),
            'learning_rate': self.reg_params.get('learning_rate', 0.01),
            'n_estimators': self.reg_params.get('n_estimators', 500),
            'subsample': self.reg_params.get('subsample', 0.6),
            'colsample_bytree': self.reg_params.get('colsample_bytree', 0.6),
            'reg_alpha': self.reg_params.get('reg_alpha', 0.1),
            'reg_lambda': self.reg_params.get('reg_lambda', 1.5),
            'min_child_weight': self.reg_params.get('min_child_weight', 10),
            'objective': 'regression', 'verbosity': -1,
        }

    def fit(self, X_train, y_cls_train, y_reg_train, X_val=None, y_cls_val=None, y_reg_val=None):
        xgb_clf_params = {
            k: v for k, v in self.clf_params.items()
            if k not in ['verbosity', 'eval_metric', 'objective', 'scale_pos_weight']
        }
        self.xgb_clf = xgb.XGBClassifier(
            **xgb_clf_params, objective='binary:logistic', eval_metric='logloss',
            scale_pos_weight=self.clf_params.get('scale_pos_weight', 1.0), verbosity=0,
        )
        fit_kw = {'verbose': False}
        if X_val is not None:
            fit_kw['eval_set'] = [(X_val, y_cls_val)]
        self.xgb_clf.fit(X_train, y_cls_train, **fit_kw)

        self.lgbm_clf = lgbm.LGBMClassifier(**self._make_lgbm_clf_params())
        fit_kw_l = {}
        if X_val is not None:
            fit_kw_l['eval_set'] = [(X_val, y_cls_val)]
            fit_kw_l['callbacks'] = [lgbm.log_evaluation(period=-1)]
        self.lgbm_clf.fit(X_train, y_cls_train, **fit_kw_l)

        xgb_reg_params = {k: v for k, v in self.reg_params.items() if k not in ['verbosity']}
        self.xgb_reg = xgb.XGBRegressor(
            **xgb_reg_params, objective='reg:squarederror', verbosity=0,
        )
        fit_kw_r = {'verbose': False}
        if X_val is not None:
            fit_kw_r['eval_set'] = [(X_val, y_reg_val)]
        self.xgb_reg.fit(X_train, y_reg_train, **fit_kw_r)

        self.lgbm_reg = lgbm.LGBMRegressor(**self._make_lgbm_reg_params())
        fit_kw_lr = {}
        if X_val is not None:
            fit_kw_lr['eval_set'] = [(X_val, y_reg_val)]
            fit_kw_lr['callbacks'] = [lgbm.log_evaluation(period=-1)]
        self.lgbm_reg.fit(X_train, y_reg_train, **fit_kw_lr)

        if X_val is not None:
            self._optimize_weights(X_val, y_cls_val, y_reg_val)

    def _optimize_weights(self, X_val, y_cls_val, y_reg_val):
        prob_xgb = self.xgb_clf.predict_proba(X_val)[:, 1]
        prob_lgbm = self.lgbm_clf.predict_proba(X_val)[:, 1]
        best_da, best_w = 0, 0.5
        for w in np.arange(0.1, 1.0, 0.05):
            blended = w * prob_xgb + (1 - w) * prob_lgbm
            da = accuracy_score(y_cls_val, (blended > 0.5).astype(int))
            if da > best_da:
                best_da, best_w = da, w
        self.clf_xgb_weight = best_w
        self.clf_lgbm_weight = 1 - best_w

        pred_xgb = self.xgb_reg.predict(X_val)
        pred_lgbm = self.lgbm_reg.predict(X_val)
        best_mae, best_rw = 999, 0.5
        for w in np.arange(0.1, 1.0, 0.05):
            blended = w * pred_xgb + (1 - w) * pred_lgbm
            mae = mean_absolute_error(y_reg_val, blended)
            if mae < best_mae:
                best_mae, best_rw = mae, w
        self.reg_xgb_weight = best_rw
        self.reg_lgbm_weight = 1 - best_rw

    def predict_proba(self, X):
        prob_xgb = self.xgb_clf.predict_proba(X)[:, 1]
        prob_lgbm = self.lgbm_clf.predict_proba(X)[:, 1]
        return self.clf_xgb_weight * prob_xgb + self.clf_lgbm_weight * prob_lgbm

    def predict_direction(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def predict_return(self, X):
        pred_xgb = self.xgb_reg.predict(X)
        pred_lgbm = self.lgbm_reg.predict(X)
        return self.reg_xgb_weight * pred_xgb + self.reg_lgbm_weight * pred_lgbm

    def predict(self, X, last_close, calibrator=None):
        raw_prob = float(self.predict_proba(X)[0])          # 분류: 상승 확률
        predicted_return = float(self.predict_return(X)[0]) # 회귀: 등락 크기

        cal_prob = calibrator.calibrate(raw_prob) if calibrator and calibrator.is_fitted else raw_prob
        direction = int(cal_prob > 0.5)

        if direction == 1 and predicted_return < 0:
            predicted_return = abs(predicted_return)
        elif direction == 0 and predicted_return > 0:
            predicted_return = -abs(predicted_return)

        predicted_price = last_close * (1 + predicted_return)

        if calibrator and calibrator.is_fitted:
            signal, confidence, bin_info = calibrator.get_dynamic_signal(cal_prob)
        else:
            confidence = abs(cal_prob - 0.5) * 2
            signal = "캘리브레이션 미적용"
            bin_info = None

        return {
            "predicted_price": round(predicted_price, 2),
            "predicted_return": round(predicted_return, 6),
            "direction": "상승" if direction == 1 else "하락",
            "raw_probability": round(raw_prob, 4),
            "calibrated_probability": round(cal_prob, 4),
            "confidence": round(confidence, 4),
            "signal": signal,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [v4 변경] Walk-Forward CV — 하이브리드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def walk_forward_cv(X, y_cls, y_reg, clf_params, reg_params, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    all_raw_probs = []
    all_y_true = []

    print(f"\n{'='*60}")
    print(f" Walk-Forward CV ({n_splits} Folds) — 하이브리드 v4 + NLP")
    print(f"{'='*60}")

    sentiment_used = [f for f in X.columns if f in SENTIMENT_FEATURES]
    if sentiment_used:
        print(f" 감성 피처: {sentiment_used}")

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_cls_tr, y_cls_te = y_cls.iloc[train_idx], y_cls.iloc[test_idx]
        y_reg_tr, y_reg_te = y_reg.iloc[train_idx], y_reg.iloc[test_idx]

        print(f"\n── Fold {fold_num}: train {len(X_train)}건 | test {len(X_test)}건")

        model = HybridEnsembleModel(clf_params, reg_params)
        model.fit(X_train, y_cls_tr, y_reg_tr, X_test, y_cls_te, y_reg_te)

        y_dir_pred = model.predict_direction(X_test)
        probs = model.predict_proba(X_test)
        y_ret_pred = model.predict_return(X_test)

        all_raw_probs.extend(probs.tolist())
        all_y_true.extend(y_cls_te.values.tolist())

        direction_acc = accuracy_score(y_cls_te, y_dir_pred)
        mae = mean_absolute_error(y_reg_te, y_ret_pred)
        rmse = np.sqrt(mean_squared_error(y_reg_te, y_ret_pred))

        metrics = {
            "Direction_Accuracy": round(direction_acc, 4),
            "MAE": round(mae, 6),
            "RMSE": round(rmse, 6),
        }
        fold_metrics.append(metrics)
        print(f"  방향성: {direction_acc:.2%} | MAE: {mae:.6f}")

    # 요약
    summary = {}
    print(f"\n{'='*60}")
    print(f" CV 요약")
    print(f"{'='*60}")

    for key in ['Direction_Accuracy', 'MAE', 'RMSE']:
        values = [m[key] for m in fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[key] = {"mean": round(mean_val, 6), "std": round(std_val, 6)}
        if key == 'Direction_Accuracy':
            print(f"  {key:>20s}: {mean_val:.2%} ± {std_val:.2%}")
        else:
            print(f"  {key:>20s}: {mean_val:.6f} ± {std_val:.6f}")

    da_values = [m['Direction_Accuracy'] for m in fold_metrics]
    da_spread = max(da_values) - min(da_values)
    print(f"  Fold간 스프레드: {da_spread:.2%}")

    # 캘리브레이션
    print(f"\n  [캘리브레이션] {len(all_raw_probs)}건으로 확률 보정 학습...")
    calibrator = ProbabilityCalibrator()
    calibrator.fit(np.array(all_raw_probs), np.array(all_y_true))

    return fold_metrics, summary, calibrator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def predict_stock_price(user_input):
    symbol = find_company_symbol(user_input)
    if not symbol:
        return None

    stock_name = get_korean_name(user_input)

    print(f"\n{'#'*60}")
    print(f"  v4: 하이브리드 + 캘리브레이션 + NLP 감성")
    print(f"  종목: {symbol} ({stock_name})")
    print(f"  피처: 기술적({len(TECH_FEATURES)}) + 매크로({len(MACRO_FEATURES)})"
          f" + 대안({len(ALTERNATIVE_FEATURES)}) + NLP({len(SENTIMENT_FEATURES)})"
          f" + 캘린더({len(CALENDAR_FEATURES)}) = {len(ALL_FEATURES)}개")
    print(f"{'#'*60}")

    # [1/6] 데이터 준비
    print("\n[1/6] 데이터 수집 + 뉴스 감성 분석...")
    df = stock_data(symbol, stock_name, years=5)
    X = df[ALL_FEATURES]
    y_cls = df['Target_Direction']
    y_reg = df['Target_Return']

    # [2/6] 피처 선택
    print("\n[2/6] 피처 중요도 분석...")
    split = int(len(X) * 0.7)
    selected_features = select_features_by_importance(
        X.iloc[:split], y_cls.iloc[:split],
        X.iloc[split:], y_cls.iloc[split:],
        threshold=0.015,
    )
    X = X[selected_features]

    # [3/6] Optuna 분류
    print(f"\n[3/6] Optuna 분류 모델 튜닝 (50회)...")
    clf_params = optimize_classifier_params(X, y_cls, n_splits=3, n_trials=50)

    # [4/6] Optuna 회귀
    print(f"\n[4/6] Optuna 회귀 모델 튜닝 (25회)...")
    reg_params = optimize_regressor_params(X, y_reg, n_splits=3, n_trials=25)

    # [5/6] Walk-Forward CV + 캘리브레이션
    print("\n[5/6] Walk-Forward CV + 확률 캘리브레이션...")
    fold_metrics, summary, calibrator = walk_forward_cv(
        X, y_cls, y_reg, clf_params, reg_params, n_splits=5,
    )

    # [6/6] 최종 예측
    print("\n[6/6] 최종 모델 학습 & 예측...")
    final_model = HybridEnsembleModel(clf_params, reg_params)
    eval_split = int(len(X) * 0.9)
    final_model.fit(
        X, y_cls, y_reg,
        X.iloc[eval_split:], y_cls.iloc[eval_split:], y_reg.iloc[eval_split:],
    )

    today_features, last_close = compare_data(symbol, stock_name, selected_features)
    result = final_model.predict(today_features, last_close, calibrator)

    print_prediction_box(symbol, result, summary)

    return {
        "symbol": symbol,
        "stock_name": stock_name,
        "last_close": last_close,
        **result,
        "cv_summary": summary,
        "fold_details": fold_metrics,
        "selected_features": selected_features,
        "clf_params": clf_params,
        "reg_params": reg_params,
        "confidence_bins": calibrator.confidence_bins if calibrator.is_fitted else None,
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 출력 포맷 함수 추가
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def print_prediction_box(symbol, result, summary):
    line = "=" * 46

    print("\n" + line)
    print(symbol)
    print(line)
    print(f"등락률:   {result['predicted_return']:+.4%}")
    print(f"방향:     {result['direction']}")
    print(f"방향성:   {summary['Direction_Accuracy']['mean']:.2%} ± {summary['Direction_Accuracy']['std']:.2%}")
    print(f"MAE:      {summary['MAE']['mean']:.6f}")
    print(line + "\n")