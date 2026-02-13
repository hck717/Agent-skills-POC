# PostgreSQL Schema Initialization Scripts

This directory contains SQL DDL scripts that automatically run when the PostgreSQL container starts for the first time.

## Execution Order

Scripts are executed in alphabetical order by the PostgreSQL Docker entrypoint:

1. **01_init_databases.sql** - Creates `equity_research` database, enables extensions, creates schemas
2. **02_quantitative_schema.sql** - Creates tables for Quantitative Fundamental Agent (coming soon)
3. **03_insider_schema.sql** - Creates TimescaleDB hypertables for Insider & Sentiment Agent (coming soon)
4. **04_shared_schema.sql** - Creates shared tables for agent logs and metadata (coming soon)

## Current Status

✅ **01_init_databases.sql** - Completed
- Creates `equity_research` database
- Enables TimescaleDB, pg_trgm, uuid-ossp extensions
- Creates schemas: `quantitative`, `insider`, `shared`
- Grants permissions to `airflow` user

⏳ **02_quantitative_schema.sql** - TODO
⏳ **03_insider_schema.sql** - TODO
⏳ **04_shared_schema.sql** - TODO

## How It Works

The `docker-compose.yml` mounts this directory:

```yaml
postgres:
  volumes:
    - ./data/postgres_schemas:/docker-entrypoint-initdb.d
```

All `.sql` and `.sh` files in `/docker-entrypoint-initdb.d/` are executed automatically on first container startup.

## Testing

To re-run initialization scripts:

```bash
# Stop and remove containers + volumes
docker-compose down -v

# Remove PostgreSQL data volume
docker volume rm postgres_data

# Restart (scripts will run again)
docker-compose up -d postgres

# Check logs
docker logs postgres
```

## Verify Installation

```bash
# Connect to PostgreSQL
docker exec -it postgres psql -U airflow -d equity_research

# Check extensions
\dx

# Check schemas
\dn

# Check TimescaleDB version
SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';
```

Expected output:
```
         extversion         
---------------------------
 2.13.0
```
