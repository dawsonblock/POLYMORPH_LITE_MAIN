#!/usr/bin/env python3
"""
Script to migrate API files from old database layer to new one.

Replaces:
- from retrofitkit.database.models import ... → from retrofitkit.db.models.* import ...
- get_session() → Depends(get_db) in function signatures
- session = get_session() → db: Session = Depends(get_db)
"""

import re
import sys
from pathlib import Path

def migrate_file(filepath: Path):
    """Migrate a single file to new DB layer."""
    print(f"Mig rating {filepath.name}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Track if we need to add get_db import
    needs_getdb_import = 'get_session()' in content
    
    # Replace get_session() calls in function bodies
    content = re.sub(
        r'(\s+)session = get_session\(\)',
        r'\1# Session will be injected via Depends(get_db)',
        content
    )
    
    # Replace audit = Audit() with audit version that takes db
    # This is trickier - for now just add a comment
    if 'audit = Audit()' in content:
        print(f"  WARNING: {filepath.name} still uses old Audit() - needs manual update")
    
    # Replace session.close() in finally blocks
    content = re.sub(
        r'finally:\s+session\.close\(\)',
        'finally:\n        pass  # Session managed by FastAPI',
        content
    )
    
    if content != original:
        # Backup
        backup_path = filepath.with_suffix('.py.bak')
        with open(backup_path, 'w') as f:
            f.write(original)
        
        # Write migrated content
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"  ✓ Migrated {filepath.name} (backup: {backup_path.name})")
        return True
    else:
        print(f"  - No changes needed for {filepath.name}")
        return False

def main():
    api_dir = Path(__file__).parent.parent / 'retrofitkit' / 'api'
    
    files_to_migrate = [
        'samples.py',
        'inventory.py',
        'calibration.py',
        'workflow_builder.py',
        'compliance.py'
    ]
    
    migrated = 0
    for filename in files_to_migrate:
        filepath = api_dir / filename
        if filepath.exists():
            if migrate_file(filepath):
                migrated += 1
        else:
            print(f"  WARNING: {filename} not found")
    
    print(f"\nMigrated {migrated} files")
    print("\nNOTE: This script made automatic changes. You still need to:")
    print("1. Add db: Session = Depends(get_db) to function signatures")
    print("2. Update Audit() calls to use db session")
    print("3. Remove get_session import and add get_db import")
    print("4. Test all endpoints")

if __name__ == '__main__':
    main()
