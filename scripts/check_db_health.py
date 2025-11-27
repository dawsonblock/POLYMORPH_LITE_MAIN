#!/usr/bin/env python3
"""
Verify database health and integrity.

Checks:
- Database connection
- All 27 tables exist
- Audit chain integrity
- RBAC roles seeded
- Sample data counts
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrofitkit.db.session import SessionLocal, get_settings
from retrofitkit.db.models.user import User
from retrofitkit.db.models.rbac import Role
from retrofitkit.db.models.sample import Sample, Project
from retrofitkit.db.models.inventory import InventoryItem
from retrofitkit.db.models.audit import AuditEvent
from retrofitkit.compliance.audit import verify_audit_chain
from sqlalchemy import inspect


EXPECTED_TABLES = [
    'organizations', 'labs', 'users', 'roles', 'user_roles',
    'audit', 'devices', 'device_status',
    'projects', 'containers', 'batches', 'samples', 'sample_lineage',
    'vendors', 'inventory_items', 'stock_lots',
    'calibration_entries',
    'workflow_versions', 'workflow_executions', 'workflow_sample_assignments', 'config_snapshots',
    'nodes', 'device_hubs'
]


def main():
    print("🔍 POLYMORPH-LITE Database Health Check")
    print("=" * 60)
    
    settings = get_settings()
    print(f"\n📍 Database: {settings.database_url.split('@')[-1] if '@' in settings.database_url else 'SQLite'}")
    
    db = SessionLocal()
    checks_passed = 0
    checks_failed = 0
    
    try:
        # Check 1: Database connection
        print("\n1️⃣  Database Connection")
        try:
            db.execute("SELECT 1")
            print("   ✅ Connected successfully")
            checks_passed += 1
        except Exception as e:
            print(f"   ❌ Connection failed: {e}")
            checks_failed += 1
            return
        
        # Check 2: Tables exist
        print("\n2️⃣  Table Schema")
        inspector = inspect(db.bind)
        existing_tables = inspector.get_table_names()
        
        missing_tables = set(EXPECTED_TABLES) - set(existing_tables)
        extra_tables = set(existing_tables) - set(EXPECTED_TABLES) - {'alembic_version'}
        
        if not missing_tables:
            print(f"   ✅ All {len(EXPECTED_TABLES)} tables exist")
            checks_passed += 1
        else:
            print(f"   ❌ Missing tables: {', '.join(missing_tables)}")
            checks_failed += 1
        
        if extra_tables:
            print(f"   ⚠️  Extra tables found: {', '.join(extra_tables)}")
        
        # Check 3: RBAC roles seeded
        print("\n3️⃣  RBAC Roles")
        roles = db.query(Role).all()
        role_names = {r.role_name for r in roles}
        expected_roles = {'admin', 'scientist', 'technician', 'compliance'}
        
        if expected_roles.issubset(role_names):
            print(f"   ✅ All default roles present: {', '.join(sorted(role_names))}")
            checks_passed += 1
        else:
            missing_roles = expected_roles - role_names
            print(f"   ❌ Missing roles: {', '.join(missing_roles)}")
            checks_failed += 1
        
        # Check 4: Audit chain integrity
        print("\n4️⃣  Audit Chain Integrity")
        audit_count = db.query(AuditEvent).count()
        if audit_count > 0:
            result = verify_audit_chain(db)
            if result['valid']:
                print(f"   ✅ Audit chain valid ({audit_count} entries)")
                checks_passed += 1
            else:
                print(f"   ❌ Audit chain broken: {len(result['errors'])} errors")
                checks_failed += 1
        else:
            print("   ⚠️  No audit entries yet (expected for new install)")
            checks_passed += 1
        
        # Check 5: Data counts
        print("\n5️⃣  Data Statistics")
        user_count = db.query(User).count()
        sample_count = db.query(Sample).count()
        project_count = db.query(Project).count()
        inventory_count = db.query(InventoryItem).count()
        
        print(f"   📊 Users: {user_count}")
        print(f"   📊 Samples: {sample_count}")
        print(f"   📊 Projects: {project_count}")
        print(f"   📊 Inventory Items: {inventory_count}")
        
        if user_count > 0:
            print("   ✅ Database has users")
            checks_passed += 1
        else:
            print("   ⚠️  No users found - run scripts/create_admin_user.py")
        
        # Summary
        print("\n" + "=" * 60)
        total_checks = checks_passed + checks_failed
        print(f"✅ Passed: {checks_passed}/{total_checks}")
        if checks_failed > 0:
            print(f"❌ Failed: {checks_failed}/{total_checks}")
            sys.exit(1)
        else:
            print("\n🎉 Database is healthy and ready!")
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db.close()


if __name__ == '__main__':
    main()
