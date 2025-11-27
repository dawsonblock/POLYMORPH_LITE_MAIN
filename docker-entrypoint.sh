#!/bin/bash
set -e

echo "🚀 POLYMORPH-LITE Docker Entrypoint"
echo "===================================="

# Simple Postgres wait (docker-compose service name)
DB_HOST="postgres"
DB_PORT="5432"

echo "⏳ Waiting for PostgreSQL at $DB_HOST:$DB_PORT..."
until pg_isready -h "$DB_HOST" -p "$DB_PORT" > /dev/null 2>&1; do
  sleep 1
done
echo "✓ PostgreSQL is ready"

# Run Alembic migrations
echo ""
echo "📦 Running database migrations..."
alembic upgrade head
echo "✓ Migrations complete"

# Seed default roles (idempotent)
echo ""
echo "👥 Seeding default roles..."
python3 -c "
from retrofitkit.db.session import SessionLocal
from retrofitkit.compliance.rbac import seed_default_roles
db = SessionLocal()
try:
    seed_default_roles(db)
    print('✓ Default roles seeded')
finally:
    db.close()
"

# Start the application
echo ""
echo "🌐 Starting POLYMORPH-LITE API server..."
echo "===================================="
exec uvicorn retrofitkit.api.server:app --host 0.0.0.0 --port 8001
