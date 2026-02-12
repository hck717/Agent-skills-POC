#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "🧪 Airflow Setup Testing Script"
echo "========================================"
echo ""

# Step 1: Check .env file
echo "📋 Step 1: Checking .env file..."
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ .env file not found!${NC}"
    echo "Please create .env file with your API keys and database credentials."
    echo "See AIRFLOW_TESTING_GUIDE.md for details."
    exit 1
fi

# Check for required variables
required_vars=("POSTGRES_URL" "NEO4J_URI" "NEO4J_USER" "NEO4J_PASSWORD" "QDRANT_URL" "QDRANT_API_KEY" "EODHD_API_KEY" "FMP_API_KEY")
missing_vars=()

for var in "${required_vars[@]}"; do
    if ! grep -q "^${var}=" .env; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Missing variables in .env:${NC}"
    for var in "${missing_vars[@]}"; do
        echo "   - $var"
    done
    echo ""
    echo "Please add these to your .env file."
    exit 1
fi

echo -e "${GREEN}✅ .env file found with all required variables${NC}"
echo ""

# Step 2: Check Docker
echo "🐳 Step 2: Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found!${NC}"
    echo "Please install Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker daemon not running!${NC}"
    echo "Please start Docker Desktop."
    exit 1
fi

echo -e "${GREEN}✅ Docker is running${NC}"
echo ""

# Step 3: Check port 8080
echo "🔌 Step 3: Checking port 8080..."
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}⚠️  Port 8080 is already in use${NC}"
    echo "If Airflow is already running, that's okay."
    echo "Otherwise, stop the process using port 8080."
    echo ""
else
    echo -e "${GREEN}✅ Port 8080 is available${NC}"
    echo ""
fi

# Step 4: Pull latest code
echo "📥 Step 4: Pulling latest code from GitHub..."
git pull origin main
echo ""

# Step 5: Start services
echo "🚀 Step 5: Starting Airflow services..."
echo "This will take 2-3 minutes on first run (installing dependencies)..."
echo ""

docker-compose down -v 2>/dev/null
docker-compose up -d

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to start services${NC}"
    echo "Check docker-compose logs for details:"
    echo "  docker-compose logs"
    exit 1
fi

echo -e "${GREEN}✅ Services started${NC}"
echo ""

# Step 6: Wait for initialization
echo "⏳ Step 6: Waiting for Airflow to initialize..."
echo "(This takes about 60-90 seconds)"
echo ""

for i in {1..90}; do
    if docker-compose logs airflow-webserver 2>/dev/null | grep -q "Airflow webserver started" || \
       curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Airflow is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""
echo ""

# Step 7: Check services status
echo "📊 Step 7: Checking services status..."
docker-compose ps
echo ""

# Step 8: Instructions
echo "========================================"
echo "🎉 Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Open Airflow UI: http://localhost:8080"
echo "   Username: admin"
echo "   Password: admin"
echo ""
echo "2. Find and run the '00_connection_test' DAG"
echo "   - Toggle it ON (if paused)"
echo "   - Click the ▶ (play) button to trigger"
echo "   - Monitor in Graph View"
echo ""
echo "3. View logs:"
echo "   docker-compose logs -f airflow-webserver"
echo "   docker-compose logs -f airflow-scheduler"
echo ""
echo "4. Check detailed guide:"
echo "   cat AIRFLOW_TESTING_GUIDE.md"
echo ""
echo "========================================"
echo "Troubleshooting:"
echo "========================================"
echo ""
echo "If connection test fails:"
echo "1. Check logs: docker-compose logs -f"
echo "2. Verify .env values are correct"
echo "3. Test cloud databases are accessible"
echo "4. See AIRFLOW_TESTING_GUIDE.md for details"
echo ""
echo "To restart everything:"
echo "  docker-compose down -v && docker-compose up -d"
echo ""
echo "========================================"
