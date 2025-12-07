#!/usr/bin/env python3
"""
Clear all data from the Emotix database.
Removes all entries from both mwb_log and raw_archive tables.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.persistence import MWBPersistence


def main():
    """Clear all data from the database."""
    parser = argparse.ArgumentParser(description='Clear all data from Emotix database')
    parser.add_argument(
        '--db',
        type=str,
        default='data/mwb_log.db',
        help='Path to database file (default: data/mwb_log.db)'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Skip confirmation prompt (use with caution!)'
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.db)
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)
    
    # Confirmation prompt
    if not args.confirm:
        print(f"⚠️  WARNING: This will delete ALL data from {db_path}")
        print("   This includes all entries from:")
        print("   - mwb_log table")
        print("   - raw_archive table")
        response = input("\n   Are you sure you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("   Cancelled.")
            sys.exit(0)
    
    # Clear database
    print(f"\n🗑️  Clearing all data from {db_path}...")
    persistence = MWBPersistence(db_path)
    result = persistence.clear_all_data()
    
    print(f"\n✓ Successfully cleared database:")
    print(f"   - Deleted {result['deleted_logs']} entries from mwb_log")
    print(f"   - Deleted {result['deleted_archive']} entries from raw_archive")
    print(f"   - Total: {result['total_deleted']} entries removed")
    print(f"\n✓ Database is now empty and ready for new data.")


if __name__ == "__main__":
    main()

