#!/bin/bash
# Validation script for POLYMORPH-LITE fixes

set -e

echo "🔍 POLYMORPH-LITE Fix Validation"
echo "================================"
echo ""

# Check Python syntax
echo "✓ Checking Python syntax..."
python3 -m py_compile retrofitkit/api/server.py
python3 -m py_compile retrofitkit/api/auth/roles.py
python3 -m py_compile retrofitkit/config.py
echo "  ✅ No syntax errors"
echo ""

# Check for duplicate definitions
echo "✓ Checking for duplicate definitions..."
LIFESPAN_COUNT=$(grep -c "^async def lifespan" retrofitkit/api/server.py || true)
if [ "$LIFESPAN_COUNT" -eq 1 ]; then
    echo "  ✅ Single lifespan definition"
else
    echo "  ❌ Found $LIFESPAN_COUNT lifespan definitions"
    exit 1
fi

APP_COUNT=$(grep -c "^app = FastAPI" retrofitkit/api/server.py || true)
if [ "$APP_COUNT" -eq 1 ]; then
    echo "  ✅ Single app creation"
else
    echo "  ❌ Found $APP_COUNT app creations"
    exit 1
fi
echo ""

# Check for deprecated on_event
echo "✓ Checking for deprecated patterns..."
if grep -q "@app.on_event" retrofitkit/api/server.py; then
    echo "  ❌ Found deprecated @app.on_event"
    exit 1
else
    echo "  ✅ No deprecated @app.on_event usage"
fi
echo ""

# Check docker-compose ports
echo "✓ Checking docker-compose configuration..."
AI_PORT=$(grep -B 5 -A 5 "ai-service:" docker-compose.yml | grep -c "3000:3000" || true)
FRONTEND_PORT=$(grep -B 5 -A 5 "frontend:" docker-compose.yml | grep -c "3001:3000" || true)
if [ "$AI_PORT" -ge 1 ] && [ "$FRONTEND_PORT" -ge 1 ]; then
    echo "  ✅ No port conflicts (AI:3000, Frontend:3001)"
else
    echo "  ⚠️  Port configuration: AI=$AI_PORT, Frontend=$FRONTEND_PORT"
fi
echo ""

# Check .env file exists
echo "✓ Checking environment configuration..."
if [ -f ".env" ]; then
    echo "  ✅ .env file exists"
else
    echo "  ⚠️  .env file missing (will use .env.example)"
fi
echo ""

# Run critical tests
echo "✓ Running critical tests..."
python3 -m pytest tests/api/test_hardening.py -q
echo "  ✅ All hardening tests pass"
echo ""

echo "================================"
echo "✅ All validations passed!"
echo ""
echo "Next steps:"
echo "  1. docker-compose up -d"
echo "  2. docker-compose exec backend python scripts/create_admin_user.py"
echo "  3. curl http://localhost:8001/health"
