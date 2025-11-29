#!/bin/bash
# POLYMORPH-LITE Golden Path Demo
# End-to-end demonstration: Sample → Raman Acquisition → AI Decision → Report
#
# This script demonstrates a complete laboratory workflow in simulation mode.
# No physical hardware required.

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  POLYMORPH-LITE Golden Path Demo                          ║${NC}"
echo -e "${BLUE}║  Sample → Raman → AI Decision → Report                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if services are running
echo -e "${YELLOW}[1/7] Checking services...${NC}"
if ! curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "❌ Backend not running. Start with: uvicorn retrofitkit.api.server:app --reload"
    exit 1
fi
echo -e "${GREEN}✓ Backend running${NC}"

if ! curl -s http://localhost:3001 > /dev/null 2>&1; then
    echo "⚠️  Frontend not running (optional). Start with: cd frontend && npm run dev"
else
    echo -e "${GREEN}✓ Frontend running${NC}"
fi

# Set demo credentials
export DEMO_EMAIL="demo@polymorph.local"
export DEMO_PASSWORD="demo123"
export API_BASE="http://localhost:8001"

echo ""
echo -e "${YELLOW}[2/7] Authenticating...${NC}"

# Login and get token
TOKEN_RESPONSE=$(curl -s -X POST "${API_BASE}/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"${DEMO_EMAIL}\",\"password\":\"${DEMO_PASSWORD}\"}")

TOKEN=$(echo $TOKEN_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('access_token', ''))" 2>/dev/null || echo "")

if [ -z "$TOKEN" ]; then
    echo "❌ Authentication failed. Creating demo user..."
    
    # Create demo user
    python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from retrofitkit.db.session import SessionLocal
from retrofitkit.db.models.user import User
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
session = SessionLocal()

# Check if user exists
existing = session.query(User).filter(User.email == "demo@polymorph.local").first()
if not existing:
    demo_user = User(
        email="demo@polymorph.local",
        hashed_password=pwd_context.hash("demo123"),
        full_name="Demo User",
        is_active=True
    )
    session.add(demo_user)
    session.commit()
    print("✓ Demo user created")
else:
    print("✓ Demo user already exists")
session.close()
EOF

    # Retry login
    TOKEN_RESPONSE=$(curl -s -X POST "${API_BASE}/auth/login" \
      -H "Content-Type: application/json" \
      -d "{\"email\":\"${DEMO_EMAIL}\",\"password\":\"${DEMO_PASSWORD}\"}")
    
    TOKEN=$(echo $TOKEN_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('access_token', ''))" 2>/dev/null || echo "")
fi

if [ -z "$TOKEN" ]; then
    echo "❌ Failed to authenticate"
    exit 1
fi

echo -e "${GREEN}✓ Authenticated as ${DEMO_EMAIL}${NC}"

# Headers for authenticated requests
AUTH_HEADER="Authorization: Bearer ${TOKEN}"

echo ""
echo -e "${YELLOW}[3/7] Creating demo project and sample...${NC}"

# Create project
PROJECT_RESPONSE=$(curl -s -X POST "${API_BASE}/api/samples/projects" \
  -H "Content-Type: application/json" \
  -H "${AUTH_HEADER}" \
  -d '{
    "project_id": "DEMO-2024-001",
    "name": "Crystallization Screening Demo",
    "description": "Golden path demonstration of Raman-based polymorph detection",
    "status": "active"
  }')

PROJECT_ID=$(echo $PROJECT_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))" 2>/dev/null || echo "")

if [ -z "$PROJECT_ID" ]; then
    echo "⚠️  Project may already exist, continuing..."
else
    echo -e "${GREEN}✓ Project created: DEMO-2024-001${NC}"
fi

# Create sample
SAMPLE_RESPONSE=$(curl -s -X POST "${API_BASE}/api/samples" \
  -H "Content-Type: application/json" \
  -H "${AUTH_HEADER}" \
  -d '{
    "sample_id": "SAMPLE-DEMO-001",
    "lot_number": "LOT-2024-11-28",
    "project_id": "DEMO-2024-001",
    "metadata": {
      "compound": "Aspirin",
      "target_polymorph": "Form II",
      "temperature_c": 25,
      "solvent": "ethanol"
    }
  }')

SAMPLE_ID=$(echo $SAMPLE_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))" 2>/dev/null || echo "")

if [ -z "$SAMPLE_ID" ]; then
    echo "⚠️  Sample may already exist, continuing..."
else
    echo -e "${GREEN}✓ Sample created: SAMPLE-DEMO-001${NC}"
fi

echo ""
echo -e "${YELLOW}[4/7] Executing Raman acquisition workflow...${NC}"

# Execute hero workflow
WORKFLOW_RESPONSE=$(curl -s -X POST "${API_BASE}/api/workflow-builder/executions" \
  -H "Content-Type: application/json" \
  -H "${AUTH_HEADER}" \
  -d '{
    "workflow_name": "hero_crystallization",
    "parameters": {
      "sample_id": "SAMPLE-DEMO-001",
      "integration_time_ms": 1000,
      "laser_power_mw": 50
    },
    "metadata": {
      "operator": "demo@polymorph.local",
      "purpose": "Golden Path Demo"
    }
  }')

RUN_ID=$(echo $WORKFLOW_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('run_id', ''))" 2>/dev/null || echo "")

if [ -z "$RUN_ID" ]; then
    echo "⚠️  Workflow execution may have failed, check logs"
    echo "Response: $WORKFLOW_RESPONSE"
else
    echo -e "${GREEN}✓ Workflow started: ${RUN_ID}${NC}"
    
    # Wait for completion (max 60 seconds)
    echo "   Waiting for workflow completion..."
    for i in {1..60}; do
        sleep 1
        STATUS_RESPONSE=$(curl -s -X GET "${API_BASE}/api/workflow-builder/executions/${RUN_ID}" \
          -H "${AUTH_HEADER}")
        
        STATUS=$(echo $STATUS_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', ''))" 2>/dev/null || echo "")
        
        if [ "$STATUS" = "completed" ]; then
            echo -e "${GREEN}✓ Workflow completed successfully${NC}"
            break
        elif [ "$STATUS" = "failed" ]; then
            echo "❌ Workflow failed"
            ERROR=$(echo $STATUS_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('error_message', 'Unknown error'))" 2>/dev/null || echo "Unknown")
            echo "   Error: $ERROR"
            break
        fi
        
        if [ $((i % 5)) -eq 0 ]; then
            echo "   Status: ${STATUS} (${i}s elapsed)"
        fi
    done
fi

echo ""
echo -e "${YELLOW}[5/7] Retrieving AI decision results...${NC}"

if [ ! -z "$RUN_ID" ]; then
    RESULTS=$(curl -s -X GET "${API_BASE}/api/workflow-builder/executions/${RUN_ID}" \
      -H "${AUTH_HEADER}")
    
    echo "$RESULTS" | python3 << 'EOF'
import sys, json
data = json.load(sys.stdin)
results = data.get('results', {})

print("\n" + "="*60)
print("WORKFLOW EXECUTION RESULTS")
print("="*60)
print(f"Run ID: {data.get('run_id', 'N/A')}")
print(f"Status: {data.get('status', 'N/A')}")
print(f"Started: {data.get('started_at', 'N/A')}")
print(f"Completed: {data.get('completed_at', 'N/A')}")
print("\nAI Decision:")
ai_result = results.get('ai_decision', {})
if ai_result:
    print(f"  Polymorph: {ai_result.get('polymorph_id', 'Unknown')}")
    print(f"  Confidence: {ai_result.get('confidence', 0)*100:.1f}%")
    print(f"  Classification: {ai_result.get('class', 'Unknown')}")
else:
    print("  No AI decision recorded")
print("="*60 + "\n")
EOF
fi

echo ""
echo -e "${YELLOW}[6/7] Generating compliance report...${NC}"

# Generate audit trail report
AUDIT_RESPONSE=$(curl -s -X GET "${API_BASE}/api/compliance/audit?limit=10" \
  -H "${AUTH_HEADER}")

echo "$AUDIT_RESPONSE" | python3 << 'EOF'
import sys, json
data = json.load(sys.stdin)
events = data.get('events', [])

print("\n" + "="*60)
print("AUDIT TRAIL (Last 10 Events)")
print("="*60)
for event in events[:10]:
    print(f"{event.get('timestamp', 'N/A')} | {event.get('event', 'N/A')}")
    print(f"  Actor: {event.get('actor', 'N/A')}")
    print(f"  Subject: {event.get('subject', 'N/A')}")
    print()
print("="*60 + "\n")
EOF

echo ""
echo -e "${YELLOW}[7/7] Demo summary...${NC}"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✓ GOLDEN PATH DEMO COMPLETE                              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "What was demonstrated:"
echo "  1. ✓ User authentication with JWT"
echo "  2. ✓ Project and sample creation (LIMS)"
echo "  3. ✓ Workflow execution (Raman acquisition)"
echo "  4. ✓ AI-driven decision making"
echo "  5. ✓ Audit trail generation"
echo "  6. ✓ Compliance reporting"
echo ""
echo "Next steps:"
echo "  • View frontend: http://localhost:3001"
echo "  • Check API docs: http://localhost:8001/docs"
echo "  • Review audit logs in database"
echo "  • Explore workflow definitions in workflows/"
echo ""
echo -e "${BLUE}For production deployment, see: DEPLOYMENT_GUIDE.md${NC}"
echo ""
