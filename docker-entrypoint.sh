#!/bin/bash
set -e

echo "🚀 POLYMORPH-LITE Docker Entrypoint"
echo "===================================="

# Function to wait for a service
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local max_attempts=30
    local attempt=0

    echo "⏳ Waiting for $service at $host:$port..."

    while [ $attempt -lt $max_attempts ]; do
        if timeout 1 bash -c "cat < /dev/null > /dev/tcp/$host/$port" 2>/dev/null; then
            echo "✓ $service is ready"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    echo "❌ $service is not available after $max_attempts attempts"
    return 1
}

# Wait for PostgreSQL (if DATABASE_URL is set)
if [ ! -z "$DATABASE_URL" ]; then
    # Extract host and port from DATABASE_URL if it's PostgreSQL
    if [[ "$DATABASE_URL" == *"postgresql"* ]]; then
        # Use default docker-compose service name
        DB_HOST="${DB_HOST:-postgres}"
        DB_PORT="${DB_PORT:-5432}"

        wait_for_service "$DB_HOST" "$DB_PORT" "PostgreSQL" || exit 1
    fi
fi

# Wait for Redis (if REDIS_HOST is set)
if [ ! -z "$REDIS_HOST" ]; then
    REDIS_PORT="${REDIS_PORT:-6379}"
    wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis" || echo "⚠️  Redis not available, continuing anyway..."
fi

# Run database migrations if enabled
if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
    echo ""
    echo "📦 Running database migrations..."
    if [ -z "${DATABASE_URL}" ]; then
        echo "❌ RUN_MIGRATIONS=true but DATABASE_URL is not set"
        exit 1
    fi
    if ALEMBIC_CONFIG="${ALEMBIC_CONFIG:-alembic.ini}" DATABASE_URL="${DATABASE_URL}" alembic -c "${ALEMBIC_CONFIG}" upgrade head; then
        echo "✓ Migrations complete"
    else
        echo "❌ Migration failed - exiting"
        exit 1
    fi
fi

# Seed default roles and data if enabled
if [ "${SEED_DATA:-true}" = "true" ]; then
    echo ""
    if [ "${SEED_DATA:-true}" = "true" ]; then
        echo ""
        if [ -z "${DATABASE_URL}" ]; then
            echo "⚠️  SEED_DATA=true but DATABASE_URL is not set; skipping seeding"
        else
            if timeout 30s python3 - <<'PYCODE'
    from retrofitkit.db.session import SessionLocal
    from retrofitkit.compliance.rbac import seed_default_roles

    db = None
    try:
        db = SessionLocal()
        seed_default_roles(db)
        print('✓ Default roles seeded')
    except Exception as e:
        print(f'⚠️  Seeding failed: {e}')
        # Do not raise to avoid crashing the container when seeding is optional
    finally:
        try:
            if db is not None:
                db.close()
        except Exception:
            pass
    PYCODE
            then
                echo "✓ Data seeding complete"
            else
                echo "⚠️  Data seeding failed, timed out, or was skipped"
            fi
        fi
    fi
fi

# Create required directories
mkdir -p /mnt/data /app/logs

# Start the application
echo ""
echo "🌐 Starting POLYMORPH-LITE API server..."
echo "Environment: ${POLYMORPH_ENV:-development}"
echo "===================================="
echo ""

# Execute the main command or the provided arguments
if [ $# -eq 0 ]; then
    # No arguments provided, use default
    exec python main.py
else
    # Execute provided command
    exec "$@"
fi
