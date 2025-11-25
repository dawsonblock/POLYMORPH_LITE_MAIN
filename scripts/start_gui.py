#!/usr/bin/env python3
"""
POLYMORPH-4 Lite GUI Startup Script
Starts both backend and frontend servers
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path

def find_project_root():
    """Find the POLYMORPH_Lite project root directory"""
    current = Path(__file__).parent
    while current.parent != current:
        if (current / "pyproject.toml").exists() and (current / "retrofitkit").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find POLYMORPH_Lite project root")

def check_node_installed():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Node.js found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("✗ Node.js not found. Please install Node.js 16+ to use the GUI")
    print("  Download from: https://nodejs.org/")
    return False

def check_npm_installed():
    """Check if npm is installed"""
    try:
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ npm found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("✗ npm not found. Please install npm")
    return False

def install_frontend_dependencies(project_root):
    """Install frontend dependencies if needed"""
    frontend_path = project_root / "gui" / "frontend"
    node_modules = frontend_path / "node_modules"
    
    if not node_modules.exists():
        print("📦 Installing frontend dependencies...")
        try:
            subprocess.run(
                ["npm", "install"], 
                cwd=frontend_path, 
                check=True
            )
            print("✓ Frontend dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install frontend dependencies: {e}")
            return False
    else:
        print("✓ Frontend dependencies already installed")
    
    return True

def install_backend_dependencies(project_root):
    """Install backend dependencies if needed"""
    gui_backend_path = project_root / "gui" / "backend"
    requirements_file = gui_backend_path / "requirements.txt"
    
    if requirements_file.exists():
        print("📦 Installing GUI backend dependencies...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
            print("✓ GUI backend dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install GUI backend dependencies: {e}")
            return False
    
    return True

def start_backend_server(project_root):
    """Start the GUI backend server"""
    gui_backend_path = project_root / "gui" / "backend"
    server_script = gui_backend_path / "gui_server.py"
    
    print("🚀 Starting GUI backend server...")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    
    try:
        process = subprocess.Popen(
            [sys.executable, str(server_script)],
            cwd=project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor backend startup
        def monitor_backend():
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"[Backend] {line.rstrip()}")
        
        backend_thread = threading.Thread(target=monitor_backend, daemon=True)
        backend_thread.start()
        
        # Wait a moment for backend to start
        time.sleep(3)
        
        if process.poll() is None:
            print("✓ GUI backend server started successfully")
            return process
        else:
            print("✗ GUI backend server failed to start")
            return None
    
    except Exception as e:
        print(f"✗ Failed to start GUI backend server: {e}")
        return None

def start_frontend_server(project_root, development=True):
    """Start the frontend development server"""
    frontend_path = project_root / "gui" / "frontend"
    
    if development:
        print("🚀 Starting frontend development server...")
        try:
            process = subprocess.Popen(
                ["npm", "start"],
                cwd=frontend_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Monitor frontend startup
            def monitor_frontend():
                for line in iter(process.stdout.readline, ''):
                    if line:
                        print(f"[Frontend] {line.rstrip()}")
            
            frontend_thread = threading.Thread(target=monitor_frontend, daemon=True)
            frontend_thread.start()
            
            print("✓ Frontend development server starting...")
            print("  Access the GUI at: http://localhost:3000")
            return process
        
        except Exception as e:
            print(f"✗ Failed to start frontend server: {e}")
            return None
    else:
        print("📦 Building frontend for production...")
        try:
            subprocess.run(["npm", "run", "build"], cwd=frontend_path, check=True)
            print("✓ Frontend built successfully")
            print("  Access the GUI at: http://localhost:8001")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to build frontend: {e}")
            return None

def main():
    """Main startup routine"""
    print("🧪 POLYMORPH-4 Lite GUI Startup")
    print("=" * 50)
    
    # Find project root
    try:
        project_root = find_project_root()
        print(f"📁 Project root: {project_root}")
    except RuntimeError as e:
        print(f"✗ {e}")
        sys.exit(1)
    
    # Check prerequisites
    if not check_node_installed() or not check_npm_installed():
        sys.exit(1)
    
    # Install dependencies
    if not install_backend_dependencies(project_root):
        sys.exit(1)
    
    if not install_frontend_dependencies(project_root):
        sys.exit(1)
    
    # Determine mode
    development = "--dev" in sys.argv or "-d" in sys.argv
    production = "--prod" in sys.argv or "-p" in sys.argv
    
    if not production:
        development = True  # Default to development mode
    
    print(f"\n🎯 Starting in {'development' if development else 'production'} mode")
    
    # Start servers
    backend_process = start_backend_server(project_root)
    if not backend_process:
        print("✗ Failed to start backend server")
        sys.exit(1)
    
    frontend_process = None
    if development:
        frontend_process = start_frontend_server(project_root, development=True)
        if not frontend_process:
            print("✗ Failed to start frontend server")
            backend_process.terminate()
            sys.exit(1)
    else:
        if not start_frontend_server(project_root, development=False):
            print("✗ Failed to build frontend")
            backend_process.terminate()
            sys.exit(1)
    
    print("\n🎉 POLYMORPH-4 Lite GUI started successfully!")
    print("\nAccess points:")
    if development:
        print("  • Development GUI: http://localhost:3000")
        print("  • Backend API: http://localhost:8001/api")
        print("  • API Documentation: http://localhost:8001/docs")
    else:
        print("  • Production GUI: http://localhost:8001")
        print("  • Backend API: http://localhost:8001/api")
        print("  • API Documentation: http://localhost:8001/docs")
    
    print("\nPress Ctrl+C to stop the servers")
    
    # Wait for interrupt
    try:
        while True:
            if backend_process and backend_process.poll() is not None:
                print("\n⚠️  Backend server stopped unexpectedly")
                break
            
            if frontend_process and frontend_process.poll() is not None:
                print("\n⚠️  Frontend server stopped unexpectedly")
                break
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
    
    # Cleanup
    if backend_process:
        backend_process.terminate()
        try:
            backend_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            backend_process.kill()
    
    if frontend_process:
        frontend_process.terminate()
        try:
            frontend_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            frontend_process.kill()
    
    print("✓ Servers stopped")

if __name__ == "__main__":
    main()