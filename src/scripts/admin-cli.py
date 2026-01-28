#!/usr/bin/env python3
"""
Admin CLI client for wisyki-api.

Allows system administrators to manage wisyki-api resources via API calls.

Usage:
    python src/scripts/admin-cli.py --help
    python src/scripts/admin-cli.py reset-collection [--host localhost] [--port 7680] [--force-recompute]
    python src/scripts/admin-cli.py status [--host localhost] [--port 7680]
"""

import sys
import argparse
import os
import requests
from typing import Optional

# Load environment variables from .env file (if present)
try:
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
except ImportError:
    pass


def reset_collection(
    host: str = "localhost",
    port: int = 7680,
    admin_key: Optional[str] = None,
    force_recompute: bool = False,
) -> bool:
    """Reset the Chroma collection via admin API."""
    if not admin_key:
        admin_key = os.getenv("ADMIN_API_KEY")
        if not admin_key:
            print("✗ Error: ADMIN_API_KEY not provided and not found in environment")
            print("  Set ADMIN_API_KEY environment variable or use --admin-key")
            return False
    
    url = f"http://{host}:{port}/admin/collection/reset"
    headers = {"X-Admin-Key": admin_key, "Content-Type": "application/json"}
    
    recompute_msg = " (forcing recomputation)" if force_recompute else ""
    print(f"Resetting collection at {host}:{port}{recompute_msg}...")
    print("⚠️  WARNING: This will delete all documents in the collection and reload from import directory")
    response = input("Are you sure? (yes/no): ").strip().lower()
    if response != "yes":
        print("Cancelled.")
        return False
    
    try:
        resp = requests.post(
            url,
            json={"confirm": True, "force_recompute": force_recompute},
            headers=headers,
            timeout=600,  # 10 minutes for large collections
        )
        
        if resp.status_code == 200:
            result = resp.json()
            print(f"\n✓ Collection reset successfully")
            print(f"  Collection: {result.get('collection')}")
            print(f"  Documents: {result.get('documents')}")
            return True
        elif resp.status_code == 401:
            print(f"✗ Authentication failed: Invalid admin key")
            return False
        else:
            print(f"✗ Error: {resp.status_code} - {resp.text}")
            return False
    
    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Could not connect to {host}:{port}")
        print("  Make sure the wisyki-api service is running")
        return False
    except requests.exceptions.Timeout:
        print(f"✗ Error: Request timed out")
        print("  Collection reset may still be in progress, check server logs")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def get_status(
    host: str = "localhost",
    port: int = 7680,
    admin_key: Optional[str] = None,
) -> bool:
    """Get collection status via admin API."""
    if not admin_key:
        admin_key = os.getenv("ADMIN_API_KEY")
        if not admin_key:
            print("✗ Error: ADMIN_API_KEY not provided and not found in environment")
            print("  Set ADMIN_API_KEY environment variable or use --admin-key")
            return False
    
    url = f"http://{host}:{port}/admin/collection/status"
    headers = {"X-Admin-Key": admin_key}
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        
        if resp.status_code == 200:
            result = resp.json()
            print(f"\n✓ Collection Status:")
            print(f"  Chroma Server: {result.get('chroma_host')}:{result.get('chroma_port')}")
            print(f"  Collection: {result.get('collection')}")
            print(f"  Documents: {result.get('documents')}")
            print(f"  Status: {result.get('status')}")
            return True
        elif resp.status_code == 401:
            print(f"✗ Authentication failed: Invalid admin key")
            return False
        else:
            print(f"✗ Error: {resp.status_code} - {resp.text}")
            return False
    
    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Could not connect to {host}:{port}")
        print("  Make sure the wisyki-api service is running")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Admin CLI client for wisyki-api",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check collection status
  python scripts/admin-cli.py status
  
  # Reset collection (with confirmation)
  python scripts/admin-cli.py reset-collection
  
  # Force recompute all embeddings (don't use precomputed)
  python scripts/admin-cli.py reset-collection --force-recompute
  
  # Use custom host/port
  python scripts/admin-cli.py status --host 192.168.1.100 --port 7680
  
  # Override admin key
  python scripts/admin-cli.py status --admin-key your-secret-key

Environment variables:
  ADMIN_API_KEY         Admin API key for authentication (required if not using --admin-key)
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Reset collection command
    reset_parser = subparsers.add_parser(
        "reset-collection",
        help="Reset and reinitialize the Chroma collection"
    )
    reset_parser.add_argument(
        "--host",
        default="localhost",
        help="wisyki-api host (default: localhost)"
    )
    reset_parser.add_argument(
        "--port",
        type=int,
        default=7680,
        help="wisyki-api port (default: 7680)"
    )
    reset_parser.add_argument(
        "--admin-key",
        help="Admin API key (default: from ADMIN_API_KEY env var)"
    )
    reset_parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recompute all embeddings, ignore precomputed ones"
    )

    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Get collection status"
    )
    status_parser.add_argument(
        "--host",
        default="localhost",
        help="wisyki-api host (default: localhost)"
    )
    status_parser.add_argument(
        "--port",
        type=int,
        default=7680,
        help="wisyki-api port (default: 7680)"
    )
    status_parser.add_argument(
        "--admin-key",
        help="Admin API key (default: from ADMIN_API_KEY env var)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "reset-collection":
        success = reset_collection(
            host=args.host,
            port=args.port,
            admin_key=args.admin_key,
            force_recompute=args.force_recompute,
        )
    elif args.command == "status":
        success = get_status(
            host=args.host,
            port=args.port,
            admin_key=args.admin_key,
        )
    else:
        parser.print_help()
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
