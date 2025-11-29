# POLYMORPH-LITE → Production LIMS Platform
## Comprehensive Upgrade Roadmap

**Version**: 2.0.0 → 3.0.0
**Target**: Enterprise Lab Operating System with LIMS, AI Orchestration, and Multi-Site Cloud
**Timeline**: 6-12 months (phased approach)

---

## 🎉 IMPLEMENTATION STATUS (Updated: 2025-11-28)

### ✅ **v3.1 Hardening Release** - **COMPLETE**

**What's Been Delivered:**

✅ **CI/CD Infrastructure** - **COMPLETE**
- Fixed all frontend path references (gui-v2/frontend → frontend)
- Updated GitHub Actions workflows (ci.yml, ci-enhanced.yml)
- Updated Dependabot configuration
- All CI jobs now correctly build and test frontend

✅ **Type Safety Improvements** - **COMPLETE**
- Stricter mypy configuration for core modules
- Type hints added to ai_client.py
- Identified 30+ remaining type errors for future work
- Core modules now enforce type safety

✅ **Backend Testing** - **COMPLETE**
- AI Circuit Breaker: 9 comprehensive tests
- Health/Metrics Endpoints: 8 tests
- All tests passing with full coverage

✅ **Frontend Testing Infrastructure** - **COMPLETE**
- Vitest configured with jsdom
- Testing Library setup complete
- Login component: 3 tests passing
- Ready for expansion to other components

✅ **Documentation** - **COMPLETE**
- Created HARDWARE_INTEGRATION.md guide
- Updated all path references across docs
- Aligned documentation with actual codebase

**Deferred to Future Release:**
- Visual Workflow Builder UI (recommended for separate feature branch)

---

### ✅ **COMPLETED** - Phases 0-4 Backend Implementation

**What's Been Delivered:**

✅ **Phase 0: Database Foundation** (Week 1-2) - **COMPLETE**
- Extended PostgreSQL schema with 20+ LIMS tables
- Alembic migrations configured and initialized
- Environment-based configuration (.env files for dev/staging/prod)
- Centralized structured logging with structlog

✅ **Phase 1: LIMS-LITE Modules** (Week 3-6) - **COMPLETE**
- Sample Tracking API with lineage (/api/samples)
- Inventory Management API with alerts (/api/inventory)
- Instrument Calibration API (/api/calibration)
- Enhanced RBAC models (database ready, API pending)

✅ **Phase 2: Workflow Builder** (Week 7-12) - **BACKEND COMPLETE**
- Visual workflow definition API (/api/workflow-builder)
- Workflow versioning with approval workflow
- Execution tracking and management
- Configuration snapshot integration

✅ **Phase 3: Compliance Package** (Week 13-18) - **COMPLETE**
- Audit trail cryptographic verification (/api/compliance)
- PDF compliance report generation
- Complete traceability matrix (sample → workflow → results)
- Configuration versioning and snapshots
- CFR 11 login enhancements (database models ready)

### 🚧 **IN PROGRESS** - Frontend UI Components

⚠️ **Remaining Work:**
- React components for sample management
- Inventory dashboard UI
- Calibration calendar view
- Visual workflow builder UI (React Flow)
- Compliance dashboard

### 📋 **PLANNED** - Phase 4: Multi-Site Cloud (Months 6-12)

❌ **Not Yet Started:**
- Cloud controller service
- Remote device hubs with gRPC
- SSO integration (Azure AD, Okta, LDAP)
- Multi-site orchestration
- Cloud admin dashboard

---

## Backend Modernization (Pydantic v2 + FastAPI)

- Migrated backend models and APIs to Pydantic v2 (`ConfigDict`, `model_config`, `model_dump()`).
- Standardized timestamps on timezone-aware UTC datetimes (`datetime.now(timezone.utc)` and `utcnow()` helpers in SQLAlchemy models).
- Replaced legacy `datetime.utcnow()` and `datetime.utcfromtimestamp()` usages in runtime code and tests.
- Clarified device registry keys for DAQ vs Raman simulators and updated orchestrator mapping to avoid collisions.
- Marked hardware-dependent tests with `pytest.mark.hardware` so they are skipped by default in CI unless explicitly enabled.

---

## Executive Summary

This roadmap transforms POLYMORPH-LITE from a sophisticated lab automation platform into a **commercially viable, investor-ready Lab Operating System** with:
- Full LIMS capabilities (Sample/Inventory/Calibration)
- Visual workflow builder with AI integration
- Enhanced 21 CFR Part 11 compliance
- Multi-site cloud orchestration

**Current State**: v2.0.0 - Production-ready automation platform
**Target State**: v3.0.0 - Enterprise LIMS with cloud orchestration

---

## Current Architecture Strengths (Preserve These!)

### ✅ Already Production-Ready
1. **Database**: PostgreSQL support implemented (`docker-compose.prod.yml`)
2. **Authentication**: JWT + MFA + bcrypt password hashing
3. **Compliance**: Audit trails, e-signatures, approval workflows
4. **AI**: BentoML integration with circuit breaker
5. **Testing**: 22 test files with pytest (unit, integration, E2E)
6. **Security**: Input validation, security headers, RBAC
7. **Orchestration**: YAML-based recipes + workflow engine
8. **Observability**: Prometheus metrics, health checks

### 🎯 Foundation for Enhancement
- SQLAlchemy ORM enables easy schema extension
- FastAPI enables rapid API development
- React 19 + Radix UI enables modern UI components
- Existing audit table provides compliance foundation

---

## PHASE 0: Database Foundation (Week 1-2)
**Status**: ⚠️ Partially Complete - PostgreSQL exists, schema needs extension

### Objectives
- Extend existing PostgreSQL schema for LIMS features
- Create Alembic migrations for version control
- Implement environment-based configuration
- Enhance logging infrastructure

### Tasks

#### 1. Schema Extension (Week 1)
**Extend existing PostgreSQL with new LIMS tables**

```sql
-- Sample Management
CREATE TABLE samples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id VARCHAR(255) UNIQUE NOT NULL,
    lot_number VARCHAR(255),
    project_id UUID REFERENCES projects(id),
    container_id UUID REFERENCES containers(id),
    parent_sample_id UUID REFERENCES samples(id),
    status VARCHAR(50) DEFAULT 'active',
    metadata JSONB,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE containers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    container_id VARCHAR(255) UNIQUE NOT NULL,
    container_type VARCHAR(100),
    location VARCHAR(255),
    capacity INTEGER,
    current_count INTEGER DEFAULT 0,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE batches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id VARCHAR(255) UNIQUE NOT NULL,
    project_id UUID REFERENCES projects(id),
    status VARCHAR(50) DEFAULT 'active',
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'active',
    owner VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Inventory Management
CREATE TABLE inventory_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    item_code VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    unit VARCHAR(50),
    min_stock INTEGER DEFAULT 0,
    current_stock INTEGER DEFAULT 0,
    location VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE stock_lots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lot_number VARCHAR(255) UNIQUE NOT NULL,
    item_id UUID REFERENCES inventory_items(id),
    vendor_id UUID REFERENCES vendors(id),
    quantity INTEGER NOT NULL,
    expiration_date DATE,
    received_date DATE DEFAULT NOW(),
    status VARCHAR(50) DEFAULT 'active',
    metadata JSONB
);

CREATE TABLE vendors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    contact_info JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Calibration Management
CREATE TABLE calibration_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id VARCHAR(255) NOT NULL,
    calibration_date TIMESTAMP NOT NULL,
    performed_by VARCHAR(255) NOT NULL,
    next_due_date DATE,
    status VARCHAR(50) DEFAULT 'valid',
    certificate_path VARCHAR(500),
    results JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE device_status (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(50) DEFAULT 'operational',
    last_calibration_date DATE,
    next_calibration_due DATE,
    health_score FLOAT,
    metadata JSONB,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Enhanced RBAC
CREATE TABLE roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    role_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    permissions JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE user_roles (
    user_email VARCHAR(255) REFERENCES users(email),
    role_id UUID REFERENCES roles(id),
    assigned_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_email, role_id)
);

-- Workflow Versioning
CREATE TABLE workflow_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_name VARCHAR(255) NOT NULL,
    version INTEGER NOT NULL,
    definition JSONB NOT NULL,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT false,
    UNIQUE (workflow_name, version)
);

-- Sample Lineage Tracking
CREATE TABLE sample_lineage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_sample_id UUID REFERENCES samples(id),
    child_sample_id UUID REFERENCES samples(id),
    relationship_type VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Add indexes for performance
CREATE INDEX idx_samples_project ON samples(project_id);
CREATE INDEX idx_samples_status ON samples(status);
CREATE INDEX idx_calibration_device ON calibration_entries(device_id);
CREATE INDEX idx_calibration_due ON calibration_entries(next_due_date);
CREATE INDEX idx_inventory_stock ON inventory_items(current_stock);
```

**Files to create:**
- `/home/user/POLYMORPH_LITE_MAIN/retrofitkit/database/models.py` - SQLAlchemy ORM models
- `/home/user/POLYMORPH_LITE_MAIN/alembic/versions/001_add_lims_tables.py` - Migration script

#### 2. Alembic Setup (Week 1)
**Initialize migration framework**

```bash
# Commands to run
alembic init alembic
alembic revision --autogenerate -m "Add LIMS tables"
alembic upgrade head
```

**Configuration files:**
- `/home/user/POLYMORPH_LITE_MAIN/alembic.ini` - Alembic config
- `/home/user/POLYMORPH_LITE_MAIN/alembic/env.py` - Migration environment

#### 3. Environment Management (Week 2)
**Enhance existing `.env` system**

**Create:**
- `.env.development` - Local development settings
- `.env.staging` - Staging environment
- `.env.production` - Production deployment (template only)

**Move all secrets to environment:**
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/polymorph_db
DATABASE_POOL_SIZE=20

# Security
SECRET_KEY=${GENERATE_RANDOM}
JWT_SECRET_KEY=${GENERATE_RANDOM}
RSA_PRIVATE_KEY_PATH=/app/secrets/private.pem

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=${GENERATE_RANDOM}

# AI Service
AI_SERVICE_URL=http://ai-service:3000
AI_CIRCUIT_BREAKER_THRESHOLD=3

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
SENTRY_DSN=${OPTIONAL}
```

**File:** `/home/user/POLYMORPH_LITE_MAIN/retrofitkit/core/config.py` (enhance existing)

#### 4. Centralized Logging (Week 2)
**Implement structured logging across all components**

```python
# retrofitkit/core/logging_config.py
import logging
import structlog

def setup_logging(log_level: str = "INFO", log_format: str = "json"):
    """Configure structured logging for all components"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if log_format == "json"
            else structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
```

**Apply to:**
- FastAPI server (`/retrofitkit/api/server.py`)
- Orchestrator (`/retrofitkit/core/orchestrator.py`)
- Device drivers (`/retrofitkit/drivers/*`)
- AI service (`/bentoml_service/service.py`)

### Deliverables - Phase 0
- [ ] PostgreSQL schema extended with 12 new LIMS tables
- [ ] Alembic migrations (initial + LIMS tables)
- [ ] Environment profiles (dev/staging/prod)
- [ ] All secrets moved to environment variables
- [ ] Centralized structured logging
- [ ] Updated `docker-compose.prod.yml` with environment injection
- [ ] Database migration tests

**Migration Command:**
```bash
alembic upgrade head  # Apply all migrations
```

---

## PHASE 1: LIMS-LITE MODULES (Week 3-6)
**Status**: ❌ New Development

### Module 1: Sample Tracking (Week 3)

#### Backend API (`/retrofitkit/api/samples.py`)

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, UUID4
from typing import List, Optional

router = APIRouter(prefix="/api/samples", tags=["samples"])

class SampleCreate(BaseModel):
    sample_id: str
    lot_number: Optional[str]
    project_id: Optional[UUID4]
    container_id: Optional[UUID4]
    metadata: dict = {}

class SampleUpdate(BaseModel):
    status: Optional[str]
    container_id: Optional[UUID4]
    metadata: Optional[dict]

class SampleResponse(BaseModel):
    id: UUID4
    sample_id: str
    lot_number: Optional[str]
    status: str
    created_by: str
    created_at: datetime
    lineage: List[dict] = []

@router.post("/", response_model=SampleResponse)
async def create_sample(sample: SampleCreate, user: User = Depends(get_current_user)):
    """Create new sample"""
    # Implementation with audit logging
    pass

@router.get("/{sample_id}", response_model=SampleResponse)
async def get_sample(sample_id: str):
    """Get sample with lineage"""
    pass

@router.put("/{sample_id}", response_model=SampleResponse)
async def update_sample(sample_id: str, update: SampleUpdate):
    """Update sample status/location"""
    pass

@router.post("/{sample_id}/assign-workflow")
async def assign_to_workflow(sample_id: str, workflow_id: str):
    """Link sample to workflow execution"""
    pass

@router.post("/{sample_id}/split")
async def split_sample(sample_id: str, child_samples: List[str]):
    """Create child samples with lineage tracking"""
    pass

@router.get("/{sample_id}/history")
async def get_sample_history(sample_id: str):
    """Get audit trail for sample"""
    pass
```

#### Frontend UI (`/frontend/src/pages/SamplesPage.tsx`)

**Components to create:**
- Sample list with search/filter
- Sample detail view with lineage graph
- Sample creation form
- Container assignment interface
- History timeline

**Features:**
- Barcode scanning support (via web camera API)
- Drag-and-drop sample → container assignment
- Real-time status updates via WebSocket

### Module 2: Inventory Tracking (Week 4)

#### Backend API (`/retrofitkit/api/inventory.py`)

```python
@router.post("/items", response_model=InventoryItemResponse)
async def create_inventory_item(item: InventoryItemCreate):
    """Add new inventory item"""
    pass

@router.get("/items/{item_code}")
async def get_inventory_item(item_code: str):
    """Get item with current stock levels"""
    pass

@router.post("/items/{item_code}/lots")
async def add_stock_lot(item_code: str, lot: StockLotCreate):
    """Add new lot to inventory"""
    pass

@router.get("/alerts/low-stock")
async def get_low_stock_alerts():
    """Get items below minimum stock"""
    pass

@router.get("/alerts/expiring")
async def get_expiring_lots(days: int = 30):
    """Get lots expiring within N days"""
    pass
```

#### Frontend UI
- Inventory dashboard with alerts
- Stock lot management
- Vendor management
- Low-stock notifications

### Module 3: Instrument Calibration (Week 5)

#### Backend API (`/retrofitkit/api/calibration.py`)

```python
@router.post("/calibration")
async def add_calibration_entry(entry: CalibrationCreate):
    """Add calibration record"""
    pass

@router.get("/calibration/device/{device_id}")
async def get_device_calibration_history(device_id: str):
    """Get all calibrations for device"""
    pass

@router.get("/calibration/upcoming")
async def get_upcoming_calibrations(days: int = 30):
    """Get devices due for calibration"""
    pass

@router.post("/calibration/{id}/attach-certificate")
async def attach_certificate(id: UUID4, file: UploadFile):
    """Upload calibration certificate PDF"""
    pass
```

#### Frontend UI
- Device calibration tab
- Calendar view for upcoming calibrations
- Certificate viewer (PDF.js integration)
- Reminder notifications

### Module 4: Enhanced RBAC (Week 6)

#### Backend Enhancement (`/retrofitkit/api/rbac.py`)

```python
# Role definitions
ROLES = {
    "Admin": ["*"],  # All permissions
    "Scientist": ["read:samples", "write:samples", "execute:workflows"],
    "Technician": ["read:samples", "execute:workflows"],
    "Compliance Officer": ["read:*", "approve:runs", "view:audit"]
}

# Permission decorator
def require_permission(permission: str):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            user = kwargs.get("user")
            if not user.has_permission(permission):
                raise HTTPException(403, "Insufficient permissions")
            return await func(*args, **kwargs)
        return wrapper
    return decorator

@router.get("/samples")
@require_permission("read:samples")
async def list_samples(user: User = Depends(get_current_user)):
    pass
```

#### Database Updates
- Migrate from flat roles to `roles` + `user_roles` tables
- Add permission sets
- Create role management API

### Deliverables - Phase 1
- [ ] Sample CRUD API with lineage tracking
- [ ] Inventory management with alerts
- [ ] Calibration logging with certificate storage
- [ ] Enhanced RBAC with granular permissions
- [ ] Frontend pages for all modules
- [ ] API tests for all endpoints
- [ ] Integration with existing audit system

---

## PHASE 2: WORKFLOW BUILDER (Week 7-12)
**Status**: ⚠️ YAML system exists, needs visual UI

### Objectives
- Create drag-and-drop workflow editor
- Enable visual workflow design without YAML
- Real-time execution preview
- Integration with existing orchestrator

### Architecture

**Existing System (Preserve):**
- YAML recipes (`/recipes/`)
- Workflow engine (`/retrofitkit/core/workflows/engine.py`)
- Orchestrator (`/retrofitkit/core/orchestrator.py`)

**New Layer:**
- Visual editor → JSON workflow definition → Orchestrator

### Frontend Implementation (Week 7-10)

#### Technology Stack
- **React Flow** - Node-based graph editor
- **Zustand** - State management
- **React Query** - Server state synchronization

#### Components (`/frontend/src/components/workflow-builder/`)

**1. WorkflowCanvas.tsx** (Week 7)
```tsx
import ReactFlow, { Node, Edge, Controls, Background } from 'reactflow';

interface WorkflowNode extends Node {
  type: 'acquire' | 'measure' | 'move' | 'gate' | 'ai-evaluate' | 'delay';
  data: {
    label: string;
    config: Record<string, any>;
  };
}

export function WorkflowCanvas() {
  const [nodes, setNodes] = useState<WorkflowNode[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      nodeTypes={nodeTypes}
    >
      <Controls />
      <Background />
    </ReactFlow>
  );
}
```

**2. Block Types** (Week 8)
- `AcquireBlock.tsx` - Raman spectrum acquisition
- `MeasureBlock.tsx` - DAQ measurement
- `MoveBlock.tsx` - Device positioning
- `GateBlock.tsx` - Conditional branching
- `AIEvaluateBlock.tsx` - AI decision node
- `DelayBlock.tsx` - Wait period

**3. Workflow Execution Viewer** (Week 9)
- Real-time step status
- Live log streaming
- Error highlighting
- Execution tree visualization

**4. Version Management** (Week 10)
- Save/load workflows
- Version history
- Diff viewer for workflow changes

### Backend Integration (Week 11-12)

#### API Endpoints (`/retrofitkit/api/workflow_builder.py`)

```python
@router.post("/workflows")
async def create_workflow(workflow: WorkflowDefinition):
    """Save workflow definition"""
    # Convert visual graph → YAML or JSON
    # Store in workflow_versions table
    pass

@router.get("/workflows/{workflow_id}/versions")
async def get_workflow_versions(workflow_id: str):
    """Get version history"""
    pass

@router.post("/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str, params: dict):
    """Execute workflow via orchestrator"""
    # Load workflow definition
    # Convert to Recipe format
    # Pass to existing orchestrator
    pass
```

#### Orchestrator Enhancement
**File:** `/retrofitkit/core/orchestrator.py`

**Add support for:**
- Loop blocks (iterate N times)
- Conditional blocks (if/else based on AI output)
- Parallel execution branches

### AI Integration

**AI Decision Node** - Automatic branching based on BentoML output

```python
# Example workflow with AI branching
{
  "type": "ai-evaluate",
  "config": {
    "service": "polymorph-detection",
    "branches": {
      "crystallizing": "continue_monitoring",
      "stable": "end_run",
      "degrading": "alert_operator"
    }
  }
}
```

### Deliverables - Phase 2
- [ ] Drag-and-drop workflow editor (React Flow)
- [ ] 6 block types implemented
- [ ] Save/load workflow definitions
- [ ] Workflow versioning system
- [ ] Real-time execution viewer
- [ ] AI decision node integration
- [ ] Loop and conditional block support
- [ ] API for workflow management
- [ ] Migration path from YAML → visual workflows

---

## PHASE 3: COMPLIANCE PACKAGE (Week 13-18)
**Status**: ⚠️ Foundation exists, needs enhancement

### Current Compliance Features (Preserve)
- ✅ Audit trails (`retrofitkit/compliance/audit.py`)
- ✅ E-signatures (`retrofitkit/compliance/signatures.py`)
- ✅ Approval workflows (`retrofitkit/compliance/approvals.py`)
- ✅ MFA login
- ✅ Password complexity enforcement

### Enhancements Required

#### 1. Hash-Chain Audit Trail (Week 13-14)

**Enhance existing `/retrofitkit/compliance/audit.py`**

```python
class EnhancedAudit:
    @staticmethod
    def create_event(event: str, actor: str, subject: str, details: dict):
        """Create audit event with cryptographic chaining"""
        # Get previous event hash
        prev_event = Audit.get_latest()
        prev_hash = prev_event.hash if prev_event else "0" * 64

        # Create new event
        timestamp = time.time()
        event_data = {
            "ts": timestamp,
            "event": event,
            "actor": actor,
            "subject": subject,
            "details": details,
            "prev_hash": prev_hash
        }

        # Calculate hash (SHA-256)
        event_json = json.dumps(event_data, sort_keys=True)
        event_hash = hashlib.sha256(event_json.encode()).hexdigest()

        # Sign with private key
        signer = Signer()
        signature = signer.sign(event_json.encode())

        # Store in database
        Audit.insert(
            ts=timestamp,
            event=event,
            actor=actor,
            subject=subject,
            details=json.dumps(details),
            prev_hash=prev_hash,
            hash=event_hash,
            signature=signature.hex(),
            public_key=signer.export_public_key().hex()
        )

        return event_hash

    @staticmethod
    def verify_chain():
        """Verify integrity of entire audit chain"""
        events = Audit.get_all_ordered()
        for i, event in enumerate(events):
            # Verify hash
            reconstructed_hash = calculate_hash(event)
            if reconstructed_hash != event.hash:
                return False, f"Hash mismatch at event {i}"

            # Verify signature
            if not verify_signature(event):
                return False, f"Signature invalid at event {i}"

            # Verify chain linkage
            if i > 0 and event.prev_hash != events[i-1].hash:
                return False, f"Chain broken at event {i}"

        return True, "Chain intact"
```

**API Endpoint:**
```python
@router.get("/audit/verify-chain")
async def verify_audit_chain():
    """Verify cryptographic integrity of audit trail"""
    is_valid, message = EnhancedAudit.verify_chain()
    return {"valid": is_valid, "message": message}
```

#### 2. E-Signature PDF Export (Week 15)

**New module:** `/retrofitkit/compliance/report_generator.py`

```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import qrcode

class ComplianceReportGenerator:
    def generate_run_report(self, run_id: str) -> bytes:
        """Generate PDF report for run with e-signatures"""
        # Load run data
        run = DataStore.load_run_metadata(run_id)
        spectra = DataStore.load_run_spectra(run_id)
        approvals = Approvals.get_by_recipe(run['recipe'])

        # Create PDF
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)

        # Header
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(50, 750, f"Run Report: {run_id}")

        # Run Details
        pdf.setFont("Helvetica", 12)
        y = 700
        pdf.drawString(50, y, f"Recipe: {run['recipe']}")
        y -= 20
        pdf.drawString(50, y, f"Operator: {run['operator']}")
        y -= 20
        pdf.drawString(50, y, f"Start Time: {datetime.fromtimestamp(run['t0'])}")

        # Data Summary
        y -= 40
        pdf.drawString(50, y, f"Total Spectra: {len(spectra)}")
        y -= 20
        pdf.drawString(50, y, f"Duration: {run['duration_sec']} seconds")

        # Approval Section
        y -= 40
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, y, "Approvals:")
        pdf.setFont("Helvetica", 10)

        for approval in approvals['approvals']:
            y -= 20
            pdf.drawString(70, y, f"{approval['email']} ({approval['role']})")
            pdf.drawString(250, y, f"{approval['timestamp']}")

            # Add signature verification QR code
            signature_data = {
                "email": approval['email'],
                "timestamp": approval['timestamp'],
                "run_id": run_id
            }
            qr = qrcode.make(json.dumps(signature_data))
            qr_img = ImageReader(qr)
            pdf.drawImage(qr_img, 450, y-15, 30, 30)

        # Audit Trail Hash
        y -= 60
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, "Audit Trail Hash:")
        pdf.setFont("Courier", 8)
        y -= 15
        audit_hash = EnhancedAudit.calculate_run_hash(run_id)
        pdf.drawString(50, y, audit_hash)

        # Footer
        pdf.setFont("Helvetica-Italic", 8)
        pdf.drawString(50, 30, f"Generated: {datetime.now()}")
        pdf.drawString(400, 30, "Page 1")

        pdf.save()
        buffer.seek(0)
        return buffer.read()
```

**API Endpoint:**
```python
@router.get("/runs/{run_id}/report.pdf")
async def download_run_report(run_id: str):
    """Download compliance report as PDF"""
    pdf_bytes = ComplianceReportGenerator().generate_run_report(run_id)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=run_{run_id}.pdf"}
    )
```

#### 3. 21 CFR Part 11 Login Flow (Week 16)

**Enhance existing `/retrofitkit/api/auth.py`**

**Requirements:**
- Account lockout after N failed attempts
- Password expiration (configurable, e.g., 90 days)
- Password history (prevent reuse of last 5 passwords)
- Session timeout (configurable)
- Concurrent session limits

```python
class CFRCompliantAuth:
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_DURATION_SEC = 1800  # 30 minutes
    PASSWORD_EXPIRY_DAYS = 90
    PASSWORD_HISTORY_COUNT = 5
    SESSION_TIMEOUT_MIN = 30

    @staticmethod
    async def login(email: str, password: str, mfa_token: Optional[str]):
        """CFR 11 compliant login"""
        # Check if account is locked
        if Users.is_locked(email):
            Audit.create_event("LOGIN_FAILED", email, "system",
                             {"reason": "account_locked"})
            raise HTTPException(403, "Account locked due to failed attempts")

        # Verify password
        user = Users.authenticate(email, password)
        if not user:
            Users.increment_failed_attempts(email)
            Audit.create_event("LOGIN_FAILED", email, "system",
                             {"reason": "invalid_credentials"})
            raise HTTPException(401, "Invalid credentials")

        # Check password expiry
        if Users.is_password_expired(email):
            Audit.create_event("LOGIN_FAILED", email, "system",
                             {"reason": "password_expired"})
            raise HTTPException(403, "Password expired - must reset")

        # Verify MFA if enabled
        if user['mfa_secret']:
            if not mfa_token:
                raise HTTPException(400, "MFA token required")
            if not Users.verify_mfa(email, mfa_token):
                Audit.create_event("LOGIN_FAILED", email, "system",
                                 {"reason": "invalid_mfa"})
                raise HTTPException(401, "Invalid MFA token")

        # Reset failed attempts
        Users.reset_failed_attempts(email)

        # Check concurrent sessions
        active_sessions = Tokens.get_active_sessions(email)
        if len(active_sessions) >= MAX_CONCURRENT_SESSIONS:
            # Revoke oldest session
            Tokens.revoke_token(active_sessions[0])

        # Create session token
        token = Tokens.create(email, user['role'])

        # Audit successful login
        Audit.create_event("LOGIN_SUCCESS", email, "system",
                         {"ip": request.client.host})

        return {"access_token": token, "token_type": "bearer"}
```

**Database additions:**
```sql
ALTER TABLE users ADD COLUMN failed_login_attempts INTEGER DEFAULT 0;
ALTER TABLE users ADD COLUMN account_locked_until TIMESTAMP;
ALTER TABLE users ADD COLUMN password_changed_at TIMESTAMP DEFAULT NOW();
ALTER TABLE users ADD COLUMN password_history TEXT[];  -- JSON array
```

#### 4. Configuration Versioning (Week 17)

**New module:** `/retrofitkit/compliance/config_versioning.py`

```python
class ConfigVersionManager:
    @staticmethod
    def snapshot_config(actor: str, reason: str):
        """Create immutable snapshot of current system config"""
        snapshot_id = str(uuid.uuid4())
        timestamp = time.time()

        # Capture all configuration
        config_data = {
            "system_config": AppContext.config.model_dump(),
            "device_registry": DeviceRegistry.export_config(),
            "user_roles": Users.export_roles(),
            "gating_rules": GatingEngine.export_rules()
        }

        # Calculate hash
        config_json = json.dumps(config_data, sort_keys=True)
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()

        # Store snapshot
        ConfigSnapshot.insert(
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            config_data=config_json,
            config_hash=config_hash,
            created_by=actor,
            reason=reason
        )

        # Audit event
        Audit.create_event("CONFIG_SNAPSHOT", actor, snapshot_id,
                         {"hash": config_hash, "reason": reason})

        return snapshot_id, config_hash
```

#### 5. Traceability Matrix Generator (Week 18)

**New API endpoint:** `/retrofitkit/api/compliance.py`

```python
@router.get("/traceability/{sample_id}")
async def generate_traceability_matrix(sample_id: str):
    """Generate complete traceability from sample → result"""
    # Get sample
    sample = Samples.get(sample_id)

    # Find all runs involving this sample
    runs = Runs.find_by_sample(sample_id)

    # Build traceability chain
    matrix = {
        "sample": {
            "id": sample_id,
            "lot": sample.lot_number,
            "created_by": sample.created_by,
            "created_at": sample.created_at
        },
        "runs": []
    }

    for run in runs:
        # Get workflow used
        workflow = Workflows.get(run.workflow_id)

        # Get approvals
        approvals = Approvals.get_by_run(run.run_id)

        # Get results
        results = DataStore.load_run_spectra(run.run_id)

        # Get config snapshot
        config = ConfigSnapshot.get(run.config_snapshot_id)

        matrix["runs"].append({
            "run_id": run.run_id,
            "workflow": {
                "name": workflow.name,
                "version": workflow.version,
                "hash": workflow.hash
            },
            "operator": run.operator,
            "timestamp": run.timestamp,
            "approvals": approvals,
            "results_count": len(results),
            "config_hash": config.config_hash,
            "audit_hash": EnhancedAudit.calculate_run_hash(run.run_id)
        })

    return matrix
```

### Deliverables - Phase 3
- [ ] Hash-chained audit trail with verification endpoint
- [ ] PDF report generation with e-signatures and QR codes
- [ ] CFR 11 compliant login flow (lockout, expiry, MFA)
- [ ] Configuration versioning system
- [ ] Traceability matrix generator (sample → run → result)
- [ ] Password history enforcement
- [ ] Session timeout and concurrent session management
- [ ] Audit chain verification UI
- [ ] Compliance dashboard showing system validation status

---

## PHASE 4: MULTI-SITE + CLOUD (Months 6-12)
**Status**: ❌ New Development - Major architectural changes

### Objectives
- Enable multi-organization/multi-lab deployment
- Cloud controller for central management
- Distributed workflow scheduling
- Remote device hubs with secure communication
- SSO/SAML/LDAP integration

### Architecture Changes

**Current (Single Site):**
```
Frontend → Backend → Database
             ↓
        Devices (local)
```

**Target (Multi-Site):**
```
Cloud Controller (SaaS)
    ↓
    ├── Lab 1 (Node + Devices)
    ├── Lab 2 (Node + Devices)
    └── Lab 3 (Node + Devices)
```

### Schema Extension (Month 6)

```sql
-- Organization hierarchy
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    subscription_tier VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE labs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lab_id VARCHAR(255) UNIQUE NOT NULL,
    organization_id UUID REFERENCES organizations(id),
    name VARCHAR(255) NOT NULL,
    location VARCHAR(255),
    timezone VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id VARCHAR(255) UNIQUE NOT NULL,
    lab_id UUID REFERENCES labs(id),
    hostname VARCHAR(255),
    ip_address VARCHAR(45),
    status VARCHAR(50) DEFAULT 'offline',
    last_heartbeat TIMESTAMP,
    capabilities JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE device_hubs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hub_id VARCHAR(255) UNIQUE NOT NULL,
    node_id UUID REFERENCES nodes(id),
    device_registry JSONB,
    health_status JSONB,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- User federation
ALTER TABLE users ADD COLUMN organization_id UUID REFERENCES organizations(id);
ALTER TABLE users ADD COLUMN lab_id UUID REFERENCES labs(id);
ALTER TABLE users ADD COLUMN sso_provider VARCHAR(100);
ALTER TABLE users ADD COLUMN sso_subject VARCHAR(255);

-- Distributed workflow scheduling
CREATE TABLE scheduled_workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id VARCHAR(255) NOT NULL,
    lab_id UUID REFERENCES labs(id),
    scheduled_time TIMESTAMP NOT NULL,
    assigned_node_id UUID REFERENCES nodes(id),
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Cloud Controller Service (Month 7-8)

**New service:** `/cloud_controller/`

```python
# cloud_controller/app.py
from fastapi import FastAPI
from fastapi.websockets import WebSocket

app = FastAPI()

class CloudController:
    """Central management service for multi-site deployment"""

    def __init__(self):
        self.connected_nodes = {}
        self.workflow_queue = PriorityQueue()

    async def register_node(self, node_id: str, capabilities: dict):
        """Register lab node with controller"""
        self.connected_nodes[node_id] = {
            "capabilities": capabilities,
            "last_heartbeat": time.time(),
            "status": "online"
        }

    async def schedule_workflow(self, workflow_id: str, lab_id: str, params: dict):
        """Schedule workflow on appropriate node"""
        # Find available node in target lab
        available_nodes = [
            n for n in self.connected_nodes.values()
            if n['lab_id'] == lab_id and n['status'] == 'online'
        ]

        if not available_nodes:
            raise ValueError(f"No online nodes in lab {lab_id}")

        # Select node (load balancing)
        selected_node = self.select_best_node(available_nodes, workflow_id)

        # Queue workflow
        await self.send_workflow_command(selected_node['node_id'], workflow_id, params)

    async def aggregate_results(self, query: dict):
        """Query results across all labs"""
        results = []
        for node_id in self.connected_nodes:
            node_results = await self.query_node(node_id, query)
            results.extend(node_results)
        return results

@app.websocket("/ws/node/{node_id}")
async def node_connection(websocket: WebSocket, node_id: str):
    """WebSocket connection for remote nodes"""
    await websocket.accept()
    await controller.register_node(node_id, await websocket.receive_json())

    try:
        while True:
            message = await websocket.receive_json()
            await controller.handle_node_message(node_id, message)
    except WebSocketDisconnect:
        controller.mark_node_offline(node_id)
```

### Remote Device Hub (Month 9)

**Enhancement to existing backend** - Add remote communication

```python
# retrofitkit/core/remote_hub.py
import grpc
from .device_hub_pb2 import DeviceCommand, DeviceResponse
from .device_hub_pb2_grpc import DeviceHubServicer

class RemoteDeviceHub(DeviceHubServicer):
    """gRPC service for remote device control"""

    def __init__(self, registry: DeviceRegistry):
        self.registry = registry
        self.cloud_controller_url = os.getenv("CLOUD_CONTROLLER_URL")
        self.node_id = os.getenv("NODE_ID")

    async def ExecuteCommand(self, request: DeviceCommand):
        """Execute device command from cloud controller"""
        device = self.registry.get(request.device_id)

        # Security: Verify command signature
        if not self.verify_command_signature(request):
            raise grpc.RpcError("Invalid command signature")

        # Execute command
        result = await device.execute(request.command, request.params)

        return DeviceResponse(
            device_id=request.device_id,
            success=True,
            data=json.dumps(result)
        )

    async def heartbeat_loop(self):
        """Send periodic heartbeat to cloud controller"""
        while True:
            await self.send_heartbeat()
            await asyncio.sleep(30)

    async def send_heartbeat(self):
        """Report node status to controller"""
        status = {
            "node_id": self.node_id,
            "timestamp": time.time(),
            "devices": self.registry.get_status_all(),
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent
        }

        async with websockets.connect(self.cloud_controller_url) as ws:
            await ws.send(json.dumps({"type": "heartbeat", "data": status}))
```

**gRPC Protocol Definition** (`/retrofitkit/core/device_hub.proto`)

```protobuf
syntax = "proto3";

service DeviceHub {
    rpc ExecuteCommand(DeviceCommand) returns (DeviceResponse);
    rpc StreamData(StreamRequest) returns (stream DataPoint);
    rpc GetStatus(StatusRequest) returns (DeviceStatus);
}

message DeviceCommand {
    string device_id = 1;
    string command = 2;
    string params = 3;  // JSON
    string signature = 4;  // HMAC signature for security
}

message DeviceResponse {
    string device_id = 1;
    bool success = 2;
    string data = 3;  // JSON
    string error = 4;
}
```

### SSO Integration (Month 10)

**New module:** `/retrofitkit/api/sso.py`

```python
from authlib.integrations.starlette_client import OAuth

oauth = OAuth()

# Configure SAML provider
oauth.register(
    name='azure',
    client_id=os.getenv('AZURE_CLIENT_ID'),
    client_secret=os.getenv('AZURE_CLIENT_SECRET'),
    server_metadata_url='https://login.microsoftonline.com/.../v2.0/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

@router.get("/auth/azure/login")
async def azure_login(request: Request):
    """Initiate Azure AD SSO login"""
    redirect_uri = request.url_for('azure_callback')
    return await oauth.azure.authorize_redirect(request, redirect_uri)

@router.get("/auth/azure/callback")
async def azure_callback(request: Request):
    """Handle Azure AD SSO callback"""
    token = await oauth.azure.authorize_access_token(request)
    user_info = await oauth.azure.parse_id_token(request, token)

    # Create or update user
    email = user_info['email']
    user = Users.get(email)
    if not user:
        # Auto-provision user from SSO
        Users.create(
            email=email,
            name=user_info['name'],
            role="Operator",  # Default role
            sso_provider="azure",
            sso_subject=user_info['sub']
        )

    # Create session token
    jwt_token = Tokens.create(email, user['role'])

    # Audit
    Audit.create_event("SSO_LOGIN", email, "azure", user_info)

    return {"access_token": jwt_token}
```

**Supported providers:**
- Azure Active Directory (SAML/OIDC)
- Okta (SAML/OIDC)
- Google Workspace (OIDC)
- Generic LDAP

### Cloud Dashboard (Month 11)

**Frontend:** `/frontend/src/pages/CloudDashboard.tsx`

**Features:**
- Real-time map of all lab locations
- Node health status (online/offline)
- Workflow execution across labs
- Aggregate metrics (total runs, active users, etc.)
- Remote device status
- Cross-lab result comparison

**Components:**
- Interactive map (Leaflet.js or Mapbox)
- Real-time WebSocket updates
- Lab selector dropdown
- Device health heatmap

### Network Security (Month 12)

**Requirements:**
- Mutual TLS (mTLS) for node ↔ controller communication
- Certificate-based authentication
- Encrypted WebSocket connections (WSS)
- IP whitelisting for cloud controller

**Certificate Management:**
```python
# cloud_controller/security/certificates.py
class CertificateAuthority:
    """Internal CA for issuing node certificates"""

    def __init__(self):
        self.ca_cert = self.load_ca_cert()
        self.ca_key = self.load_ca_key()

    def issue_node_certificate(self, node_id: str, lab_id: str):
        """Issue certificate for new node"""
        # Generate CSR
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )

        # Create certificate
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, f"node-{node_id}"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, lab_id)
        ])

        cert = x509.CertificateBuilder() \
            .subject_name(subject) \
            .issuer_name(self.ca_cert.subject) \
            .public_key(key.public_key()) \
            .serial_number(x509.random_serial_number()) \
            .not_valid_before(datetime.now(timezone.utc)) \
            .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365)) \
            .sign(self.ca_key, hashes.SHA256())

        return cert, key
```
- [ ] Multi-tenant database schema (organizations, labs, nodes)
- [ ] Cloud controller service with WebSocket communication
- [ ] Remote device hub with gRPC
- [ ] SSO integration (Azure AD, Okta, LDAP)
- [ ] Distributed workflow scheduler
- [ ] Cloud dashboard UI
- [ ] mTLS certificate management
- [ ] Multi-lab result aggregation
- [ ] Site-level RBAC
- [ ] Deployment guide for cloud controller
- [ ] Kubernetes Helm charts for cloud deployment

---

## DEPLOYMENT STRATEGY

### Development Environment
```bash
# Local development (existing)
docker-compose up
```

### Staging Environment
```bash
# Staging with PostgreSQL
docker-compose -f docker-compose.prod.yml up
```

### Production Deployment (Cloud)

**Kubernetes Helm Chart** (`/helm/polymorph-cloud/`)

```yaml
# values.yaml
cloudController:
  replicas: 3
  image: polymorph/cloud-controller:3.0.0
  database:
    host: postgres.default.svc.cluster.local
    port: 5432
  redis:
    host: redis.default.svc.cluster.local

labNode:
  image: polymorph/lab-node:3.0.0
  cloudControllerUrl: wss://cloud.polymorph.example.com/ws/node
  devices:
    - type: raman
      driver: ocean_optics
    - type: daq
      driver: ni_daq
```

**Deploy:**
```bash
helm install polymorph-cloud ./helm/polymorph-cloud \
  --set cloudController.database.password=$DB_PASSWORD \
  --set cloudController.jwt.secret=$JWT_SECRET
```

---

## TESTING STRATEGY

### Unit Tests (All Phases)
- Test all new models, APIs, and utilities
- Maintain >80% code coverage
- Run on every commit (CI/CD)

### Integration Tests (Phase 1, 2, 3)
- Test LIMS workflows end-to-end
- Test workflow builder → orchestrator integration
- Test compliance reporting pipeline

### E2E Tests (Phase 2, 4)
- Selenium/Playwright tests for frontend
- Test complete user journeys (sample → run → report)
- Test multi-site workflow execution

### Performance Tests (Phase 4)
- Load test cloud controller (1000 concurrent nodes)
- Test database query performance with 1M+ audit records
- WebSocket stress testing

### Security Tests (Phase 3, 4)
- Penetration testing for authentication
- Audit trail integrity verification
- Certificate validation testing

---

## DOCUMENTATION DELIVERABLES

### Technical Documentation
- [ ] API Reference (OpenAPI/Swagger)
- [ ] Database schema documentation
- [ ] Deployment guides (Docker, Kubernetes)
- [ ] Developer onboarding guide

### User Documentation
- [ ] User manual for LIMS features
- [ ] Workflow builder tutorial
- [ ] Compliance guide (21 CFR Part 11)
- [ ] Administrator handbook

### Compliance Documentation
- [ ] Validation protocols (IQ/OQ/PQ templates)
- [ ] Traceability matrix
- [ ] Security architecture document
- [ ] Audit trail specification

---

## RESOURCE REQUIREMENTS

### Development Team
- **Backend Engineers**: 2-3 (Python/FastAPI/PostgreSQL)
- **Frontend Engineers**: 1-2 (React/TypeScript)
- **DevOps Engineer**: 1 (Docker/Kubernetes/Helm)
- **QA Engineer**: 1 (Test automation)
- **Compliance Specialist**: 1 (Part 11 validation)

### Infrastructure
- **Development**: 2x servers (16GB RAM, 8 cores)
- **Staging**: 1x server (32GB RAM, 16 cores)
- **Production Cloud**: Kubernetes cluster (3+ nodes)
- **Database**: PostgreSQL 15+ (managed service recommended)

### Timeline
- **Phase 0**: 2 weeks
- **Phase 1**: 4 weeks
- **Phase 2**: 6 weeks
- **Phase 3**: 6 weeks
- **Phase 4**: 6 months

**Total**: 9-12 months for full implementation

---

## SUCCESS METRICS

### Technical Metrics
- [ ] 100% test coverage for compliance features
- [ ] <100ms API response time (P95)
- [ ] 99.9% uptime for cloud controller
- [ ] Support 1000+ concurrent devices

### Business Metrics
- [ ] Enable multi-tenant SaaS deployment
- [ ] Support pharma industry compliance requirements
- [ ] Enable enterprise contracts (multi-site)
- [ ] Reduce deployment time to <1 hour (Helm)

### User Metrics
- [ ] <5 minute onboarding for new users
- [ ] <30 second workflow creation (visual builder)
- [ ] Zero manual SQL queries for operators

---

## RISK MITIGATION

### Technical Risks
| Risk | Mitigation |
|------|-----------|
| Database migration breaks existing runs | Comprehensive backup + rollback plan, migration testing |
| Cloud controller single point of failure | High availability (3+ replicas), health checks |
| Network latency for remote devices | Local caching, graceful degradation |
| Certificate expiration | Automated renewal (cert-manager on K8s) |

### Compliance Risks
| Risk | Mitigation |
|------|-----------|
| Audit trail tampering | Cryptographic hash chaining, immutable storage |
| Unauthorized access | mTLS, SSO, MFA enforcement |
| Data loss | Automated backups (PostgreSQL WAL archiving) |

### Operational Risks
| Risk | Mitigation |
|------|-----------|
| Complex deployment | Helm charts, automated CI/CD |
| Developer onboarding | Comprehensive documentation |
| Breaking changes | Semantic versioning, API versioning |

---

## NEXT STEPS

1. **Review this roadmap** with stakeholders
2. **Prioritize phases** based on business needs
3. **Allocate resources** (team, budget, infrastructure)
4. **Set up project tracking** (Jira, Linear, GitHub Projects)
5. **Begin Phase 0** implementation

---

## APPENDIX: File Structure (Post-Upgrade)

```
POLYMORPH_LITE_MAIN/
├── retrofitkit/
│   ├── api/
│   │   ├── samples.py              # NEW - Sample API
│   │   ├── inventory.py            # NEW - Inventory API
│   │   ├── calibration.py          # NEW - Calibration API
│   │   ├── workflow_builder.py     # NEW - Workflow builder API
│   │   ├── rbac.py                 # NEW - Enhanced RBAC
│   │   └── compliance.py           # NEW - Compliance endpoints
│   ├── compliance/
│   │   ├── report_generator.py     # NEW - PDF generation
│   │   └── config_versioning.py    # NEW - Config snapshots
│   ├── database/
│   │   └── models.py               # ENHANCED - LIMS models
│   └── core/
│       └── remote_hub.py           # NEW - Remote device hub
├── cloud_controller/               # NEW - Cloud service
│   ├── app.py
│   ├── scheduler.py
│   └── security/
├── frontend/src/
│   ├── pages/
│   │   ├── SamplesPage.tsx         # NEW
│   │   ├── InventoryPage.tsx       # NEW
│   │   ├── CalibrationPage.tsx     # NEW
│   │   ├── WorkflowBuilderPage.tsx # NEW
│   │   └── CloudDashboard.tsx      # NEW
│   └── components/
│       └── workflow-builder/       # NEW - React Flow components
├── alembic/                        # NEW - Database migrations
│   └── versions/
├── helm/                           # NEW - Kubernetes deployment
│   └── polymorph-cloud/
├── docs/                           # ENHANCED - Documentation
│   ├── api/
│   ├── user-manual/
│   └── compliance/
└── tests/
    ├── test_lims.py               # NEW - LIMS tests
    ├── test_workflow_builder.py   # NEW
    └── test_multi_site.py         # NEW
```

---

**Document Version**: 1.0
**Last Updated**: 2025-11-27
**Author**: POLYMORPH Engineering Team
**Status**: Ready for Implementation
