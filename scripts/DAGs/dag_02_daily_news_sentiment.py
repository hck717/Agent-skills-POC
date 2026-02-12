"""
DAG 02: Daily News Sentiment Pipeline
Schedule: 3x daily (7 AM, 12 PM, 6 PM HKT)
Purpose: Ingest breaking news, sentiment, and SEC filing alerts
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
import requests
import psycopg2
import feedparser
import re

import sys
sys.path.insert(0, '/opt/airflow/dags')
from config import (
    DEFAULT_ARGS, DEFAULT_TICKERS, TEST_MODE,
    POSTGRES_URL, EODHD_API_KEY, FMP_API_KEY,
    EODHD_BASE_URL, FMP_BASE_URL, MAX_ACTIVE_RUNS, MAX_NEWS_ARTICLES
)

logger = logging.getLogger(__name__)

def extract_fmp_latest_news(**context):
    """Extract last 8 hours of news from FMP"""
    logger.info(f"Extracting FMP news for {len(DEFAULT_TICKERS)} tickers")
    news_data = []
    
    for ticker in DEFAULT_TICKERS:
        try:
            url = f"{FMP_BASE_URL}/stock_news"
            params = {"tickers": ticker, "limit": MAX_NEWS_ARTICLES, "apikey": FMP_API_KEY}
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            articles = response.json()
            
            for article in articles:
                news_data.append({
                    'ticker': ticker,
                    'headline': article.get('title', ''),
                    'summary': article.get('text', '')[:500],
                    'publish_date': article.get('publishedDate'),
                    'source': article.get('site', 'fmp'),
                    'url': article.get('url', ''),
                    'sentiment_score': article.get('sentiment', 0)
                })
            
            logger.info(f"✅ {ticker}: {len(articles)} articles")
        except Exception as e:
            logger.warning(f"FMP news failed for {ticker}: {str(e)}")
    
    context['ti'].xcom_push(key='news_data', value=news_data)
    return {"extracted": len(news_data)}

def extract_eodhd_news(**context):
    """Extract EODHD news feed as backup"""
    news_data = context['ti'].xcom_pull(task_ids='extract_fmp_latest_news', key='news_data') or []
    logger.info("Extracting EODHD news feed")
    
    try:
        url = f"{EODHD_BASE_URL}/news"
        params = {"api_token": EODHD_API_KEY, "fmt": "json", "limit": MAX_NEWS_ARTICLES * len(DEFAULT_TICKERS)}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        articles = response.json()
        
        for article in articles:
            tags = article.get('tags', [])
            relevant_tickers = [t for t in tags if t in DEFAULT_TICKERS]
            
            for ticker in relevant_tickers:
                news_data.append({
                    'ticker': ticker,
                    'headline': article.get('title', ''),
                    'summary': article.get('content', '')[:500],
                    'publish_date': article.get('date'),
                    'source': 'eodhd',
                    'url': article.get('link', ''),
                    'sentiment_score': article.get('sentiment', 0)
                })
        
        logger.info(f"✅ EODHD: {len([n for n in news_data if n['source']=='eodhd'])} articles")
    except Exception as e:
        logger.warning(f"EODHD news failed: {str(e)}")
    
    context['ti'].xcom_push(key='news_data', value=news_data)
    return {"extracted": len(news_data)}

def extract_sec_edgar_rss(**context):
    """Monitor SEC EDGAR RSS for new filings"""
    logger.info("Monitoring SEC EDGAR RSS feed")
    filings_data = []
    
    try:
        rss_url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK=&type=&company=&dateb=&owner=include&start=0&count=100&output=atom"
        feed = feedparser.parse(rss_url)
        
        for entry in feed.entries:
            title = entry.get('title', '')
            for ticker in DEFAULT_TICKERS:
                if ticker in title.upper():
                    form_match = re.search(r'(\d+-[KQ]|8-K)', title)
                    form_type = form_match.group(1) if form_match else 'UNKNOWN'
                    
                    filings_data.append({
                        'ticker': ticker,
                        'form_type': form_type,
                        'filing_date': entry.get('published', ''),
                        'title': title,
                        'link': entry.get('link', ''),
                        'summary': entry.get('summary', '')[:200]
                    })
        
        logger.info(f"✅ SEC EDGAR: {len(filings_data)} new filings")
    except Exception as e:
        logger.warning(f"SEC EDGAR RSS failed: {str(e)}")
    
    context['ti'].xcom_push(key='filings_data', value=filings_data)
    return {"extracted": len(filings_data)}

def transform_news_sentiment(**context):
    """Deduplicate and extract entities from news"""
    news_data = context['ti'].xcom_pull(task_ids='extract_eodhd_news', key='news_data') or []
    logger.info(f"Transforming {len(news_data)} news articles")
    
    import pandas as pd
    df = pd.DataFrame(news_data)
    
    if df.empty:
        context['ti'].xcom_push(key='transformed_news', value=[])
        return {"transformed": 0}
    
    # Deduplicate by headline similarity (simple exact match for now)
    df = df.drop_duplicates(subset=['headline', 'ticker'])
    
    # Extract publish timestamp
    df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
    
    # Filter last 24 hours
    cutoff = datetime.now() - timedelta(hours=24)
    df = df[df['publish_date'] >= cutoff]
    
    # Sort by publish date
    df = df.sort_values('publish_date', ascending=False)
    
    transformed_data = df.to_dict('records')
    context['ti'].xcom_push(key='transformed_news', value=transformed_data)
    logger.info(f"✅ Transformed: {len(transformed_data)} articles (deduplicated)")
    return {"transformed": len(transformed_data)}

def load_postgres_news(**context):
    """Load news to Postgres with TimescaleDB"""
    transformed_news = context['ti'].xcom_pull(task_ids='transform_news_sentiment', key='transformed_news') or []
    filings_data = context['ti'].xcom_pull(task_ids='extract_sec_edgar_rss', key='filings_data') or []
    
    if not transformed_news and not filings_data:
        logger.info("No news or filings to load")
        return {"loaded": 0}
    
    logger.info(f"Loading {len(transformed_news)} news articles and {len(filings_data)} filings to Postgres")
    
    conn = psycopg2.connect(POSTGRES_URL)
    cursor = conn.cursor()
    
    # Create news_articles table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10),
            headline TEXT,
            summary TEXT,
            publish_date TIMESTAMP,
            source VARCHAR(50),
            url TEXT,
            sentiment_score NUMERIC,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, headline, publish_date)
        )
    """)
    
    # Create sec_filings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sec_filings (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10),
            form_type VARCHAR(20),
            filing_date TIMESTAMP,
            title TEXT,
            link TEXT,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, form_type, filing_date)
        )
    """)
    
    # UPSERT news articles
    for article in transformed_news:
        cursor.execute("""
            INSERT INTO news_articles (ticker, headline, summary, publish_date, source, url, sentiment_score)
            VALUES (%(ticker)s, %(headline)s, %(summary)s, %(publish_date)s, %(source)s, %(url)s, %(sentiment_score)s)
            ON CONFLICT (ticker, headline, publish_date) DO UPDATE SET
                summary = EXCLUDED.summary,
                sentiment_score = EXCLUDED.sentiment_score
        """, article)
    
    # UPSERT SEC filings
    for filing in filings_data:
        cursor.execute("""
            INSERT INTO sec_filings (ticker, form_type, filing_date, title, link, summary)
            VALUES (%(ticker)s, %(form_type)s, %(filing_date)s, %(title)s, %(link)s, %(summary)s)
            ON CONFLICT (ticker, form_type, filing_date) DO NOTHING
        """, filing)
    
    conn.commit()
    cursor.close()
    conn.close()
    
    logger.info(f"✅ Loaded {len(transformed_news)} news + {len(filings_data)} filings to Postgres")
    return {"loaded": len(transformed_news) + len(filings_data)}

# DAG DEFINITION
with DAG(
    dag_id='02_daily_news_sentiment_pipeline',
    default_args=DEFAULT_ARGS,
    description='3x daily news and sentiment ingestion',
    schedule_interval='0 */6 * * *' if not TEST_MODE else None,  # Every 6 hours
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=MAX_ACTIVE_RUNS,
    tags=['daily', 'news', 'sentiment'],
) as dag:
    
    task_extract_fmp = PythonOperator(task_id='extract_fmp_latest_news', python_callable=extract_fmp_latest_news)
    task_extract_eodhd = PythonOperator(task_id='extract_eodhd_news', python_callable=extract_eodhd_news)
    task_extract_sec = PythonOperator(task_id='extract_sec_edgar_rss', python_callable=extract_sec_edgar_rss)
    task_transform = PythonOperator(task_id='transform_news_sentiment', python_callable=transform_news_sentiment)
    task_load_postgres = PythonOperator(task_id='load_postgres_news', python_callable=load_postgres_news)
    
    task_extract_fmp >> task_extract_eodhd >> task_transform
    task_extract_sec >> task_load_postgres
    task_transform >> task_load_postgres
