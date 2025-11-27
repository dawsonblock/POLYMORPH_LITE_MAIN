#!/usr/bin/env python3
"""
Create initial admin user for POLYMORPH-LITE.

Usage:
    python scripts/create_admin_user.py
    
Or with custom credentials:
    python scripts/create_admin_user.py --email admin@example.com --name "Admin User" --password yourpassword
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrofitkit.db.session import SessionLocal
from retrofitkit.compliance.users import create_user
from retrofitkit.compliance.rbac import seed_default_roles, assign_role


def main():
    parser = argparse.ArgumentParser(description='Create admin user for POLYMORPH-LITE')
    parser.add_argument('--email', default='admin@polymorph.local', help='Admin email')
    parser.add_argument('--name', default='System Administrator', help='Admin name')
    parser.add_argument('--password', default='admin123', help='Admin password (change immediately!)')
    
    args = parser.parse_args()
    
    print("🔧 POLYMORPH-LITE Admin User Setup")
    print("=" * 50)
    
    db = SessionLocal()
    
    try:
        # Seed default roles
        print("\n1. Seeding default roles...")
        seed_default_roles(db)
        print("   ✓ Roles created: admin, scientist, technician, compliance")
        
        # Create admin user
        print(f"\n2. Creating admin user: {args.email}")
        user = create_user(
            db=db,
            email=args.email,
            password=args.password,
            full_name=args.name,
            role="admin",  # Legacy role field
            is_superuser=True
        )
        print(f"   ✓ User created: {user.email}")
        
        # Assign admin role via RBAC
        print(f"\n3. Assigning admin role...")
        assign_role(db, args.email, "admin", assigned_by="system")
        print(f"   ✓ Role assigned")
        
        print("\n" + "=" * 50)
        print("✅ Admin user created successfully!")
        print("\nCredentials:")
        print(f"   Email:    {args.email}")
        print(f"   Password: {args.password}")
        print("\n⚠️  IMPORTANT: Change this password immediately in production!")
        print(f"\nLogin at: http://localhost:8001/auth/login")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        db.rollback()
        sys.exit(1)
    finally:
        db.close()


if __name__ == '__main__':
    main()
