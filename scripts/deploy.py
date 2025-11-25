#!/usr/bin/env python3
"""
Polymorph-4 Deployment Script
Handles production deployment with Docker
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
import shutil

def run_command(cmd, cwd=None):
    """Run shell command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {cmd}")
        if result.stdout:
            print(f"  {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {cmd}")
        print(f"Error: {e.stderr}")
        return False

def create_env_file():
    """Create .env file from template if it doesn't exist"""
    env_file = Path("docker/.env")
    env_template = Path("docker/.env.example")
    
    if not env_file.exists() and env_template.exists():
        print("Creating .env file from template...")
        shutil.copy(env_template, env_file)
        print("✓ Created docker/.env")
        print("⚠️  Please edit docker/.env with your production settings")
        return False  # Need manual configuration
    return True

def deploy_development():
    """Deploy development environment"""
    print("\n=== Deploying Development Environment ===")
    
    # Use explicit compose file path from repo root
    if not run_command("docker compose -f docker/docker-compose.yml up --build -d"):
        return False
    
    print("\n🚀 Development environment deployed!")
    print("📊 Access points:")
    print("  • Main App: http://localhost:8000")
    print("  • API Docs: http://localhost:8000/docs")
    
    return True

def deploy_production():
    """Deploy production environment with observability"""
    print("\n=== Deploying Production Environment ===")
    
    # Check if .env file exists
    if not create_env_file():
        return False
    
    # Deploy with production compose file
    cmd = "docker compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml -f docker/docker-compose.observability.yml up --build -d"
    if not run_command(cmd):
        return False
    
    print("\n🚀 Production environment deployed!")
    print("📊 Access points:")
    print("  • Main App: http://localhost:80 (or your domain)")
    print("  • Grafana: http://localhost:3000 (admin/admin)")
    print("  • Prometheus: http://localhost:9090")
    
    return True

def deploy_hardware():
    """Deploy hardware-enabled environment"""
    print("\n=== Deploying Hardware Environment ===")
    
    # Build hardware image
    if not run_command("docker build -f Dockerfile.multi --target hardware -t polymorph4:hardware ."):
        return False
    
    # Run with hardware support
    cmd = """docker run -d \
        --name polymorph4-hardware \
        --privileged \
        --device=/dev/bus/usb \
        -v /dev:/dev \
        -p 8000:8000 \
        -v polymorph4_data:/app/data \
        -v polymorph4_logs:/app/logs \
        polymorph4:hardware"""
    
    if not run_command(cmd):
        return False
    
    print("\n🚀 Hardware environment deployed!")
    print("📊 Access points:")
    print("  • Main App: http://localhost:8000")
    print("⚠️  Hardware devices should be accessible in container")
    
    return True

def status():
    """Show deployment status"""
    print("\n=== Deployment Status ===")
    
    # Show status using the base compose file context
    run_command("docker compose -f docker/docker-compose.yml ps")
    
    print("\n📊 Service URLs:")
    print("  • Main App: http://localhost:8000")
    print("  • Grafana: http://localhost:3000")
    print("  • Prometheus: http://localhost:9090")

def stop_services():
    """Stop all services"""
    print("\n=== Stopping Services ===")
    
    # Stop compose services
    run_command("docker compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml -f docker/docker-compose.observability.yml down")
    
    # Stop hardware container if running
    run_command("docker stop polymorph4-hardware")
    run_command("docker rm polymorph4-hardware")
    
    print("✓ All services stopped")

def cleanup():
    """Clean up Docker resources"""
    print("\n=== Cleanup ===")
    
    stop_services()
    
    # Remove images
    run_command("docker image prune -f")
    
    # Remove volumes (careful!)
    print("⚠️  This will delete all data volumes!")
    if input("Continue? (y/N): ").lower().startswith('y'):
        run_command("docker volume prune -f")
        print("✓ Cleanup completed")
    else:
        print("Cleanup cancelled")

def main():
    parser = argparse.ArgumentParser(description="Polymorph-4 Deployment Script")
    parser.add_argument("command", choices=["dev", "prod", "hardware", "status", "stop", "cleanup"],
                       help="Deployment command")
    
    args = parser.parse_args()
    
    print("🐳 Polymorph-4 Docker Deployment")
    print("=" * 40)
    
    success = True
    
    if args.command == "dev":
        success = deploy_development()
    elif args.command == "prod":
        success = deploy_production()
    elif args.command == "hardware":
        success = deploy_hardware()
    elif args.command == "status":
        status()
    elif args.command == "stop":
        stop_services()
    elif args.command == "cleanup":
        cleanup()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()