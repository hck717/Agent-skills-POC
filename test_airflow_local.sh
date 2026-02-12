#!/bin/bash
# Test Airflow DAGs locally without Docker
# Usage: ./test_airflow_local.sh

set -e  # Exit on error

echo "🚀 Setting up local Airflow environment..."

# Load .env file
if [ -f .env ]; then
    echo "✅ Loading .env file..."
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "❌ .env file not found!"
    exit 1
fi

# Set Airflow-specific variables
export AIRFLOW_HOME=$(pwd)
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/scripts/DAGs
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__EXECUTOR=SequentialExecutor
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="${POSTGRES_URL}"
export AIRFLOW__CORE__FERNET_KEY="81HqDtbqAywKSOumSha3BhWNOdQ26slT6K0YaZeZyPs="

echo "✅ Environment variables loaded"
echo "   AIRFLOW_HOME: ${AIRFLOW_HOME}"
echo "   DAGS_FOLDER: ${AIRFLOW__CORE__DAGS_FOLDER}"
echo "   TEST_MODE: ${TEST_MODE}"
echo ""

# Check if virtual environment exists
if [ ! -d "venv_airflow" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv_airflow
    source venv_airflow/bin/activate
    
    echo "📦 Installing Airflow and dependencies..."
    pip install -q --upgrade pip
    pip install -q "apache-airflow==2.8.1" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.1/constraints-3.11.txt"
    pip install -q psycopg2-binary neo4j qdrant-client requests pandas
    echo "✅ Installation complete"
else
    source venv_airflow/bin/activate
    echo "✅ Using existing virtual environment"
fi

echo ""
echo "🔧 Initializing Airflow database..."
airflow db migrate 2>&1 | grep -E "(INFO|ERROR|WARNING)" | tail -5

echo ""
echo "👤 Creating admin user (if not exists)..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin 2>/dev/null || echo "✅ Admin user already exists"

echo ""
echo "="*70
echo "🧪 TESTING DAG: 00_connection_test"
echo "="*70
echo ""

# Test the connection test DAG
airflow dags test 00_connection_test 2026-02-12

echo ""
echo "="*70
echo "✅ Local test complete!"
echo "="*70
echo ""
echo "To test individual tasks:"
echo "  source venv_airflow/bin/activate"
echo "  export \$(cat .env | grep -v '^#' | xargs)"
echo "  export AIRFLOW_HOME=\$(pwd)"
echo "  airflow tasks test 00_connection_test test_postgres 2026-02-12"
echo "  airflow tasks test 00_connection_test test_neo4j 2026-02-12"
echo "  airflow tasks test 00_connection_test test_qdrant 2026-02-12"
echo "  airflow tasks test 00_connection_test test_api_keys 2026-02-12"
echo ""
