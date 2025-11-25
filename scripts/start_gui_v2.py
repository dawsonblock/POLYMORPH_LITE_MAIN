#!/usr/bin/env python3
"""
POLYMORPH-4 Lite GUI v2.0 Smart Startup Script
Ultra-modern startup with dependency management and health checks
"""

import os
import sys
import subprocess
import time
import signal
import threading
import argparse
import webbrowser
from pathlib import Path
from typing import Optional, Dict, Any
import json

# Rich console for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    from rich.live import Live
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    def rprint(*args, **kwargs):
        print(*args, **kwargs)

console = Console() if RICH_AVAILABLE else None

def find_project_root():
    """Find the POLYMORPH_Lite project root directory"""
    current = Path(__file__).parent.parent
    if (current / "gui-v2").exists():
        return current
    raise RuntimeError("Could not find project root with gui-v2 directory")

def check_system_requirements():
    """Check system requirements for modern development"""
    requirements = {
        "node": {"min_version": "18.0.0", "found": False, "version": None},
        "npm": {"min_version": "8.0.0", "found": False, "version": None},
        "python": {"min_version": "3.9.0", "found": False, "version": None},
    }
    
    # Check Node.js
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip().lstrip('v')
            requirements["node"]["version"] = version
            requirements["node"]["found"] = True
    except FileNotFoundError:
        pass
    
    # Check npm
    try:
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            requirements["npm"]["version"] = version
            requirements["npm"]["found"] = True
    except FileNotFoundError:
        pass
    
    # Check Python
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    requirements["python"]["version"] = version
    requirements["python"]["found"] = True
    
    return requirements

def display_system_check(requirements: Dict[str, Any]):
    """Display system requirements check"""
    if not RICH_AVAILABLE:
        print("=== System Requirements Check ===")
        for tool, info in requirements.items():
            status = "✅" if info["found"] else "❌"
            print(f"{status} {tool}: {info['version'] or 'Not found'}")
        return
    
    table = Table(title="🔍 System Requirements Check")
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Required", style="yellow")
    table.add_column("Found", style="green")
    table.add_column("Status", justify="center")
    
    for tool, info in requirements.items():
        status = "✅" if info["found"] else "❌"
        table.add_row(
            tool.capitalize(),
            f">= {info['min_version']}",
            info["version"] or "Not found",
            status
        )
    
    console.print(table)

def install_dependencies(project_root: Path, mode: str = "dev"):
    """Install dependencies with progress tracking"""
    frontend_path = project_root / "gui-v2" / "frontend"
    backend_path = project_root / "gui-v2" / "backend"
    
    if not RICH_AVAILABLE:
        print("📦 Installing dependencies...")
        
        # Frontend dependencies
        print("Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_path, check=True)
        
        # Backend dependencies
        print("Installing backend dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], cwd=backend_path, check=True)
        
        print("✅ Dependencies installed successfully")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Install frontend dependencies
        frontend_task = progress.add_task("Installing frontend dependencies...", total=None)
        try:
            subprocess.run(
                ["npm", "install"], 
                cwd=frontend_path, 
                check=True,
                capture_output=True
            )
            progress.update(frontend_task, description="✅ Frontend dependencies installed")
        except subprocess.CalledProcessError as e:
            progress.update(frontend_task, description="❌ Frontend dependencies failed")
            raise
        
        # Install backend dependencies
        backend_task = progress.add_task("Installing backend dependencies...", total=None)
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], cwd=backend_path, check=True, capture_output=True)
            progress.update(backend_task, description="✅ Backend dependencies installed")
        except subprocess.CalledProcessError as e:
            progress.update(backend_task, description="❌ Backend dependencies failed")
            raise

def start_backend(project_root: Path, mode: str = "dev"):
    """Start the backend server"""
    backend_path = project_root / "gui-v2" / "backend"
    server_script = backend_path / "main.py"
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    env["ENVIRONMENT"] = mode
    
    if mode == "dev":
        cmd = [sys.executable, str(server_script)]
    else:
        cmd = [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
    
    process = subprocess.Popen(
        cmd,
        cwd=backend_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    return process

def start_frontend(project_root: Path, mode: str = "dev"):
    """Start the frontend development server"""
    frontend_path = project_root / "gui-v2" / "frontend"
    
    if mode == "dev":
        cmd = ["npm", "run", "dev"]
    else:
        # First build, then serve
        subprocess.run(["npm", "run", "build"], cwd=frontend_path, check=True)
        cmd = ["npm", "run", "preview"]
    
    process = subprocess.Popen(
        cmd,
        cwd=frontend_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    return process

def monitor_services(backend_process, frontend_process):
    """Monitor both services and display logs"""
    if not RICH_AVAILABLE:
        print("🚀 Services started! Monitoring logs...")
        print("Backend: http://localhost:8000")
        print("Frontend: http://localhost:3000")
        print("Press Ctrl+C to stop all services")
        
        def print_logs(process, prefix):
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"[{prefix}] {line.rstrip()}")
        
        backend_thread = threading.Thread(
            target=print_logs, 
            args=(backend_process, "Backend"), 
            daemon=True
        )
        frontend_thread = threading.Thread(
            target=print_logs, 
            args=(frontend_process, "Frontend"), 
            daemon=True
        )
        
        backend_thread.start()
        frontend_thread.start()
        
        return
    
    # Rich console monitoring
    console.print(Panel.fit(
        "[bold green]🚀 POLYMORPH-4 Lite GUI v2.0 Started![/bold green]\n\n"
        "[cyan]Backend:[/cyan] http://localhost:8001\n"
        "[cyan]Frontend:[/cyan] http://localhost:3000\n"
        "[cyan]API Docs:[/cyan] http://localhost:8001/api/docs\n\n"
        "[yellow]Press Ctrl+C to stop all services[/yellow]",
        title="Services Running"
    ))

def wait_for_services():
    """Wait for services to be ready and open browser"""
    import requests
    
    # Wait for backend
    for _ in range(30):
        try:
            requests.get("http://localhost:8001/api/health", timeout=1)
            break
        except:
            time.sleep(1)
    
    # Wait for frontend
    time.sleep(3)
    
    # Open browser
    try:
        webbrowser.open("http://localhost:3000")
        if RICH_AVAILABLE:
            console.print("🌐 [green]Browser opened to http://localhost:3000[/green]")
    except:
        pass

def cleanup_processes(*processes):
    """Clean up all processes"""
    for process in processes:
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass

def main():
    parser = argparse.ArgumentParser(description="POLYMORPH-4 Lite GUI v2.0 Startup Script")
    parser.add_argument("--mode", choices=["dev", "prod"], default="dev", 
                       help="Run in development or production mode")
    parser.add_argument("--skip-install", action="store_true", 
                       help="Skip dependency installation")
    parser.add_argument("--no-browser", action="store_true",
                       help="Don't automatically open browser")
    
    args = parser.parse_args()
    
    try:
        # Display header
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold blue]POLYMORPH-4 Lite GUI v2.0[/bold blue]\n"
                "[cyan]Ultra-Modern Analytical Instrument Automation[/cyan]",
                title="🧪 Startup Script"
            ))
        else:
            print("🧪 POLYMORPH-4 Lite GUI v2.0 - Ultra-Modern Edition")
            print("=" * 60)
        
        # Check system requirements
        requirements = check_system_requirements()
        display_system_check(requirements)
        
        # Validate requirements
        missing = [tool for tool, info in requirements.items() if not info["found"]]
        if missing:
            if RICH_AVAILABLE:
                console.print(f"[red]❌ Missing required tools: {', '.join(missing)}[/red]")
            else:
                print(f"❌ Missing required tools: {', '.join(missing)}")
            return 1
        
        # Find project root
        project_root = find_project_root()
        
        # Install dependencies
        if not args.skip_install:
            install_dependencies(project_root, args.mode)
        
        # Start services
        if RICH_AVAILABLE:
            console.print("\n🚀 [bold green]Starting services...[/bold green]")
        else:
            print("\n🚀 Starting services...")
        
        backend_process = start_backend(project_root, args.mode)
        time.sleep(2)  # Let backend start first
        
        frontend_process = start_frontend(project_root, args.mode)
        time.sleep(2)  # Let frontend start
        
        # Wait for services and open browser
        if not args.no_browser:
            threading.Thread(target=wait_for_services, daemon=True).start()
        
        # Monitor services
        monitor_services(backend_process, frontend_process)
        
        # Keep running until interrupted
        try:
            while True:
                if backend_process.poll() is not None or frontend_process.poll() is not None:
                    if RICH_AVAILABLE:
                        console.print("[red]❌ A service has stopped unexpectedly[/red]")
                    else:
                        print("❌ A service has stopped unexpectedly")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            if RICH_AVAILABLE:
                console.print("\n[yellow]🛑 Shutting down services...[/yellow]")
            else:
                print("\n🛑 Shutting down services...")
        
        # Cleanup
        cleanup_processes(backend_process, frontend_process)
        
        if RICH_AVAILABLE:
            console.print("[green]✅ All services stopped cleanly[/green]")
        else:
            print("✅ All services stopped cleanly")
        
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())