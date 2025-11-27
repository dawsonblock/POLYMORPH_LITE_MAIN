#!/usr/bin/env python3
"""
Backup PostgreSQL database to compressed file.

Usage:
    python scripts/backup_database.py
    python scripts/backup_database.py --output /path/to/backup.sql.gz
"""

import argparse
import subprocess
import os
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrofitkit.db.session import get_settings


def main():
    parser = argparse.ArgumentParser(description='Backup POLYMORPH-LITE database')
    parser.add_argument('--output', help='Output file path', default=None)
    parser.add_argument('--no-compress', action='store_true', help='Skip compression')
    
    args = parser.parse_args()
    
    settings = get_settings()
    
    # Parse DATABASE_URL
    if 'postgresql' not in settings.database_url:
        print("❌ This script only supports PostgreSQL databases")
        sys.exit(1)
    
    # Extract connection info from DATABASE_URL
    # Format: postgresql+psycopg2://user:pass@host:port/dbname
    url = settings.database_url.replace('postgresql+psycopg2://', 'postgresql://')
    
    # Generate filename if not provided
    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = os.path.join('backups')
        os.makedirs(backup_dir, exist_ok=True)
        
        if args.no_compress:
            args.output = os.path.join(backup_dir, f'polymorph_backup_{timestamp}.sql')
        else:
            args.output = os.path.join(backup_dir, f'polymorph_backup_{timestamp}.sql.gz')
    
    print(f"🗄️  POLYMORPH-LITE Database Backup")
    print(f"📁 Output: {args.output}")
    print("=" * 60)
    
    try:
        # Run pg_dump
        if args.no_compress:
            cmd = ['pg_dump', url, '-f', args.output, '--clean', '--if-exists']
        else:
            cmd = f'pg_dump "{url}" --clean --if-exists | gzip > "{args.output}"'
        
        print("⏳ Running backup...")
        
        if args.no_compress:
            result = subprocess.run(cmd, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            file_size = os.path.getsize(args.output) / (1024 * 1024)  # MB
            print(f"✅ Backup completed successfully")
            print(f"📊 Size: {file_size:.2f} MB")
            print(f"📂 Location: {os.path.abspath(args.output)}")
        else:
            print(f"❌ Backup failed:")
            print(result.stderr)
            sys.exit(1)
            
    except FileNotFoundError:
        print("❌ pg_dump not found. Please install PostgreSQL client tools.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
