"""
DAG 2: Daily News & Sentiment Pipeline
Schedule: 3x daily (7 AM, 12 PM, 6 PM HKT)
Purpose: Ingest breaking news, sentiment, and SEC filings
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import config

# ============================================================================
# TASK FUNCTIONS
# ============================================================================

def extract_fmp_latest_news(**context):
    """
    Fetch last 8 hours of news articles from FMP
    """
    import requests
    import pandas as pd
    from datetime import datetime, timedelta
    
    execution_time = datetime.fromisoformat(context['ts'])
    lookback_hours = 8
    from_time = (execution_time - timedelta(hours=lookback_hours)).strftime("%Y-%m-%dT%H:%M:%S")
    
    all_news = []
    
    # General market news
    url = f"{config.FMP_BASE_URL}/stock_news"
    params = {
        "apikey": config.FMP_API_KEY,
        "limit": 100
    }
    
    try:
        response = requests.get(url, params=params, timeout=config.API_TIMEOUT_SECONDS)
        response.raise_for_status()
        news_data = response.json()
        all_news.extend(news_data)
    except Exception as e:
        print(f"Error fetching general news: {e}")
    
    # Ticker-specific news
    for ticker in config.DEFAULT_TICKERS[:10]:  # Limit to avoid rate limits
        url = f"{config.FMP_BASE_URL}/stock_news"
        params = {
            "tickers": ticker,
            "limit": 10,
            "apikey": config.FMP_API_KEY
        }
        
        try:
            response = requests.get(url, params=params, timeout=config.API_TIMEOUT_SECONDS)
            response.raise_for_status()
            ticker_news = response.json()
            all_news.extend(ticker_news)
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
    
    context['ti'].xcom_push(key='fmp_news', value=all_news)
    print(f"Extracted {len(all_news)} news articles from FMP")
    return True


def extract_sec_edgar_rss(**context):
    """
    Monitor SEC EDGAR RSS for new filings
    """
    import feedparser
    from datetime import datetime, timedelta
    
    execution_time = datetime.fromisoformat(context['ts'])
    lookback_hours = 8
    cutoff_time = execution_time - timedelta(hours=lookback_hours)
    
    # SEC EDGAR RSS feed
    rss_url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK=&type=&company=&dateb=&owner=include&start=0&count=100&output=atom"
    
    try:
        feed = feedparser.parse(rss_url)
        
        recent_filings = []
        for entry in feed.entries:
            filing_date = datetime(*entry.published_parsed[:6])
            
            if filing_date >= cutoff_time:
                # Parse filing info
                filing_info = {
                    'title': entry.title,
                    'link': entry.link,
                    'published': entry.published,
                    'summary': entry.summary,
                    'filing_date': filing_date.isoformat()
                }
                
                # Extract form type (10-K, 10-Q, 8-K)
                if '10-K' in entry.title:
                    filing_info['form_type'] = '10-K'
                    recent_filings.append(filing_info)
                elif '10-Q' in entry.title:
                    filing_info['form_type'] = '10-Q'
                    recent_filings.append(filing_info)
                elif '8-K' in entry.title:
                    filing_info['form_type'] = '8-K'
                    recent_filings.append(filing_info)
        
        context['ti'].xcom_push(key='sec_filings', value=recent_filings)
        print(f"Found {len(recent_filings)} recent SEC filings")
        
        # Trigger quarterly SEC filings DAG if 10-K or 10-Q found
        if any(f['form_type'] in ['10-K', '10-Q'] for f in recent_filings):
            context['ti'].xcom_push(key='trigger_sec_dag', value=True)
        
        return True
        
    except Exception as e:
        print(f"Error fetching SEC EDGAR RSS: {e}")
        return False


def transform_news_sentiment(**context):
    """
    Deduplicate, sentiment analysis, entity extraction
    """
    import pandas as pd
    from difflib import SequenceMatcher
    
    ti = context['ti']
    fmp_news = ti.xcom_pull(key='fmp_news', task_ids='extract_fmp_latest_news')
    
    if not fmp_news:
        print("No news to process")
        return False
    
    df = pd.DataFrame(fmp_news)
    
    # Deduplicate by title similarity
    unique_news = []
    seen_titles = []
    
    for _, row in df.iterrows():
        title = row.get('title', '')
        is_duplicate = False
        
        for seen_title in seen_titles:
            similarity = SequenceMatcher(None, title.lower(), seen_title.lower()).ratio()
            if similarity > 0.85:  # 85% similarity threshold
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_news.append(row.to_dict())
            seen_titles.append(title)
    
    # Sentiment analysis placeholder
    # In production: Use FinBERT or FMP's sentiment scores
    for article in unique_news:
        article['sentiment_score'] = article.get('sentiment', 0)  # -1 to 1
    
    # Entity extraction placeholder
    # In production: Use NER model to extract tickers, companies, people
    for article in unique_news:
        article['extracted_tickers'] = []  # Would extract from text
    
    context['ti'].xcom_push(key='processed_news', value=unique_news)
    print(f"Processed {len(unique_news)} unique news articles")
    return True


def load_postgres_news(**context):
    """
    Load news into Postgres with TimescaleDB
    """
    import psycopg2
    from psycopg2.extras import execute_values
    import json
    
    ti = context['ti']
    processed_news = ti.xcom_pull(key='processed_news', task_ids='transform_news_sentiment')
    
    if not processed_news:
        print("No news to load")
        return False
    
    # Connect to Postgres
    conn = psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        dbname=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD
    )
    cur = conn.cursor()
    
    # Create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            summary TEXT,
            url TEXT,
            source VARCHAR(100),
            publish_date TIMESTAMP NOT NULL,
            sentiment_score FLOAT,
            extracted_tickers TEXT[],
            raw_data JSONB,
            ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create TimescaleDB hypertable (if not already)
    try:
        cur.execute("""
            SELECT create_hypertable('news_articles', 'publish_date', 
                if_not_exists => TRUE, migrate_data => TRUE)
        """)
    except Exception as e:
        print(f"Hypertable creation note: {e}")
    
    # Insert news articles (upsert based on URL)
    insert_query = """
        INSERT INTO news_articles (title, summary, url, source, publish_date, sentiment_score, extracted_tickers, raw_data)
        VALUES %s
        ON CONFLICT (url) DO UPDATE SET
            sentiment_score = EXCLUDED.sentiment_score,
            extracted_tickers = EXCLUDED.extracted_tickers
    """
    
    values = [
        (
            article.get('title'),
            article.get('text', article.get('summary')),
            article.get('url', article.get('link')),
            article.get('site', 'FMP'),
            article.get('publishedDate', article.get('published')),
            article.get('sentiment_score', 0.0),
            article.get('extracted_tickers', []),
            json.dumps(article)
        )
        for article in processed_news
    ]
    
    execute_values(cur, insert_query, values)
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"Loaded {len(processed_news)} news articles into Postgres")
    return True


def check_trigger_sec_dag(**context):
    """
    Check if SEC filings DAG should be triggered
    """
    ti = context['ti']
    trigger_flag = ti.xcom_pull(key='trigger_sec_dag', task_ids='extract_sec_edgar_rss')
    
    if trigger_flag:
        print("New 10-K/10-Q filings detected - will trigger DAG 4")
        return True
    else:
        print("No new filings requiring processing")
        return False


# ============================================================================
# DAG DEFINITION
# ============================================================================

default_args = config.DEFAULT_ARGS.copy()
default_args.update({
    "start_date": datetime(2026, 1, 1),
})

with DAG(
    dag_id="dag_02_daily_news_sentiment_pipeline",
    default_args=default_args,
    description="Ingest breaking news, sentiment, and SEC filings",
    schedule_interval="0 7,12,18 * * *",  # 7 AM, 12 PM, 6 PM HKT
    catchup=False,
    max_active_runs=config.MAX_ACTIVE_RUNS,
    tags=["daily", "news", "sentiment", "3x_daily"],
) as dag:

    # Extract tasks
    task_extract_fmp_news = PythonOperator(
        task_id="extract_fmp_latest_news",
        python_callable=extract_fmp_latest_news,
    )

    task_extract_sec = PythonOperator(
        task_id="extract_sec_edgar_rss",
        python_callable=extract_sec_edgar_rss,
    )

    # Transform task
    task_transform = PythonOperator(
        task_id="transform_news_sentiment",
        python_callable=transform_news_sentiment,
    )

    # Load task
    task_load_news = PythonOperator(
        task_id="load_postgres_news",
        python_callable=load_postgres_news,
    )

    # Conditional trigger for SEC filings DAG
    task_check_trigger = PythonOperator(
        task_id="check_trigger_sec_dag",
        python_callable=check_trigger_sec_dag,
    )

    task_trigger_sec_dag = TriggerDagRunOperator(
        task_id="trigger_quarterly_sec_filings",
        trigger_dag_id="dag_04_quarterly_sec_filings_pipeline",
        wait_for_completion=False,
    )

    # Define dependencies
    [task_extract_fmp_news, task_extract_sec] >> task_transform
    task_transform >> task_load_news
    task_extract_sec >> task_check_trigger >> task_trigger_sec_dag
