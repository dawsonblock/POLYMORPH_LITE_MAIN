#!/usr/bin/env python3
"""
Unified Polymorph-4 Installation & Setup Script
Combines base installation, hardware configuration, and overlay application
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run shell command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {cmd}")
        print(f"Error: {e.stderr}")
        return False

def install_dependencies(hardware=False):
    """Install Python dependencies"""
    print("\n=== Installing Dependencies ===")
    
    # Basic dependencies
    if not run_command("pip install --upgrade pip"):
        return False
    
    if not run_command("pip install -r requirements.txt"):
        return False
    
    # Hardware dependencies
    if hardware:
        print("\nInstalling hardware dependencies...")
        if not run_command("pip install -r requirements-hw.txt"):
            print("Warning: Hardware dependencies failed to install")
            print("This is normal if hardware drivers are not available")
    
    return True

def initialize_system(admin_email, admin_name):
    """Initialize the system with admin user"""
    print("\n=== Initializing System ===")
    
    cmd = f'python -m retrofitkit.cli init --admin-email "{admin_email}" --admin-name "{admin_name}" --set-admin-password'
    return run_command(cmd)

def configure_hardware():
    """Run hardware wizard for automatic configuration"""
    print("\n=== Hardware Configuration ===")
    print("Starting hardware wizard...")
    
    try:
        subprocess.run([sys.executable, "scripts/hardware_wizard.py"], check=True)
        return True
    except subprocess.CalledProcessError:
        print("Hardware wizard failed or was cancelled")
        return False

def apply_overlay(overlay_name):
    """Apply a configuration overlay"""
    print(f"\n=== Applying Overlay: {overlay_name} ===")
    
    overlay_path = f"config/overlays/{overlay_name}"
    if not os.path.exists(overlay_path):
        print(f"Error: Overlay '{overlay_name}' not found")
        available = [d for d in os.listdir("config/overlays") if os.path.isdir(f"config/overlays/{d}")]
        print(f"Available overlays: {', '.join(available)}")
        return False
    
    cmd = f'python scripts/apply_overlay.py {overlay_name} .'
    return run_command(cmd)

def start_server(host="0.0.0.0", port=8000):
    """Start the development server"""
    print(f"\n=== Starting Server ===")
    print(f"Server will be available at: http://{host}:{port}")
    print("Press Ctrl+C to stop")
    
    cmd = f"uvicorn retrofitkit.api.server:app --host {host} --port {port} --reload"
    os.system(cmd)

def main():
    parser = argparse.ArgumentParser(description="Polymorph-4 Unified Installation & Setup")
    parser.add_argument("--admin-email", default="admin@local", help="Admin email")
    parser.add_argument("--admin-name", default="Admin", help="Admin name")
    parser.add_argument("--hardware", action="store_true", help="Install hardware dependencies")
    parser.add_argument("--configure-hardware", action="store_true", help="Run hardware wizard")
    parser.add_argument("--overlay", help="Apply configuration overlay")
    parser.add_argument("--start-server", action="store_true", help="Start development server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--full-setup", action="store_true", help="Complete setup workflow")
    
    args = parser.parse_args()
    
    print("🔬 Polymorph-4 Unified Installation & Setup")
    print("=" * 50)
    
    # Full setup workflow
    if args.full_setup:
        print("Starting full setup workflow...")
        
        # 1. Install dependencies
        if not install_dependencies(hardware=True):
            sys.exit(1)
        
        # 2. Initialize system
        if not initialize_system(args.admin_email, args.admin_name):
            sys.exit(1)
        
        # 3. Hardware configuration
        print("\nWould you like to configure hardware automatically? (y/n): ", end="")
        if input().lower().startswith('y'):
            configure_hardware()
        
        # 4. Apply overlay
        print("\nAvailable configuration overlays:")
        overlays = [d for d in os.listdir("config/overlays") if os.path.isdir(f"config/overlays/{d}")]
        for i, overlay in enumerate(overlays):
            print(f"  [{i}] {overlay}")
        
        print("Enter overlay number (or skip): ", end="")
        choice = input().strip()
        if choice.isdigit() and 0 <= int(choice) < len(overlays):
            apply_overlay(overlays[int(choice)])
        
        # 5. Start server
        print("\nWould you like to start the server now? (y/n): ", end="")
        if input().lower().startswith('y'):
            start_server(args.host, args.port)
        
        return
    
    # Individual steps
    success = True
    
    if args.hardware or not any([args.configure_hardware, args.overlay, args.start_server]):
        success &= install_dependencies(hardware=args.hardware)
        success &= initialize_system(args.admin_email, args.admin_name)
    
    if args.configure_hardware:
        configure_hardware()
    
    if args.overlay:
        success &= apply_overlay(args.overlay)
    
    if args.start_server:
        if success:
            start_server(args.host, args.port)
        else:
            print("Not starting server due to previous errors")
            sys.exit(1)

if __name__ == "__main__":
    main()