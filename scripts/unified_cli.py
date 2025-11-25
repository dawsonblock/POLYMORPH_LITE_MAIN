#!/usr/bin/env python3
"""
Polymorph-4 Unified CLI
Provides integrated access to all system functions
"""
import os
import sys
import subprocess
import typer
from typing import Optional
from pathlib import Path

app = typer.Typer(name="polymorph4", help="Polymorph-4 Unified Control System")

# Hardware management commands
hardware_app = typer.Typer(name="hardware", help="Hardware configuration commands")
app.add_typer(hardware_app, name="hardware")

# Configuration management commands  
config_app = typer.Typer(name="config", help="Configuration management commands")
app.add_typer(config_app, name="config")

# System management commands
system_app = typer.Typer(name="system", help="System management commands") 
app.add_typer(system_app, name="system")

def run_script(script_path: str, *args):
    """Run a script with arguments"""
    cmd = [sys.executable, script_path] + list(args)
    return subprocess.run(cmd, cwd=Path(__file__).parent.parent)

@app.command()
def install(
    hardware: bool = typer.Option(False, "--hardware", "-h", help="Install hardware dependencies"),
    full: bool = typer.Option(False, "--full", "-f", help="Full interactive setup")
):
    """Install and setup the system"""
    args = []
    if hardware:
        args.append("--hardware")
    if full:
        args.append("--full-setup")
    
    return run_script("install.py", *args)

@app.command()
def server(
    host: str = typer.Option("0.0.0.0", help="Host address"),
    port: int = typer.Option(8000, help="Port number"),
    reload: bool = typer.Option(True, help="Enable auto-reload")
):
    """Start the development server"""
    cmd = f"uvicorn retrofitkit.api.server:app --host {host} --port {port}"
    if reload:
        cmd += " --reload"
    
    typer.echo(f"Starting server at http://{host}:{port}")
    os.system(cmd)

@hardware_app.command("wizard")
def hardware_wizard():
    """Run interactive hardware configuration wizard"""
    typer.echo("🔧 Starting hardware configuration wizard...")
    return run_script("scripts/hardware_wizard.py")

@hardware_app.command("list")
def hardware_list():
    """List detected hardware devices"""
    typer.echo("🔍 Detecting hardware...")
    
    # Try to detect NI devices
    try:
        import nidaqmx
        from nidaqmx.system import System
        sys_local = System.local()
        devices = list(sys_local.devices)
        if devices:
            typer.echo("\n📟 NI-DAQmx Devices:")
            for dev in devices:
                typer.echo(f"  • {dev.name} ({dev.product_type})")
        else:
            typer.echo("  No NI devices found")
    except ImportError:
        typer.echo("  NI-DAQmx not installed")
    except Exception as e:
        typer.echo(f"  NI detection error: {e}")
    
    # Try to detect Ocean devices  
    try:
        from seabreeze.spectrometers import list_devices
        devices = list_devices()
        if devices:
            typer.echo("\n🌊 Ocean Optics Devices:")
            for i, dev in enumerate(devices):
                typer.echo(f"  • [{i}] {dev}")
        else:
            typer.echo("  No Ocean Optics devices found")
    except ImportError:
        typer.echo("  SeaBreeze not installed")
    except Exception as e:
        typer.echo(f"  Ocean detection error: {e}")

@hardware_app.command("profile")
def hardware_profile(
    profile: str = typer.Argument(help="Hardware profile name (e.g., ni_usb_6343)")
):
    """Apply a hardware profile"""
    profile_path = f"config/hardware_profiles/{profile}.yaml"
    if not os.path.exists(profile_path):
        typer.echo(f"❌ Profile not found: {profile_path}")
        typer.echo("Available profiles:")
        profiles_dir = Path("config/hardware_profiles")
        if profiles_dir.exists():
            for p in profiles_dir.glob("*.yaml"):
                typer.echo(f"  • {p.stem}")
        typer.raise_typer.Exit(1)
    
    typer.echo(f"📋 Applying hardware profile: {profile}")
    return run_script("scripts/select_hardware_profile.py", profile_path)

@config_app.command("overlay")
def config_overlay(
    overlay: str = typer.Argument(help="Overlay name"),
    target: str = typer.Option(".", help="Target directory")
):
    """Apply a configuration overlay"""
    overlay_path = f"config/overlays/{overlay}"
    if not os.path.exists(overlay_path):
        typer.echo(f"❌ Overlay not found: {overlay}")
        typer.echo("Available overlays:")
        overlays_dir = Path("config/overlays")
        if overlays_dir.exists():
            for d in overlays_dir.iterdir():
                if d.is_dir():
                    typer.echo(f"  • {d.name}")
        typer.raise_typer.Exit(1)
    
    typer.echo(f"⚙️  Applying configuration overlay: {overlay}")
    return run_script("scripts/apply_overlay.py", overlay, target)

@config_app.command("list")
def config_list():
    """List available configuration overlays and profiles"""
    typer.echo("📁 Configuration Options:")
    
    # List overlays
    overlays_dir = Path("config/overlays")
    if overlays_dir.exists():
        typer.echo("\n🎛️  Available Overlays:")
        for d in overlays_dir.iterdir():
            if d.is_dir():
                # Try to read description from config
                config_file = d / "config.yaml"
                desc = ""
                if config_file.exists():
                    import yaml
                    try:
                        with open(config_file) as f:
                            cfg = yaml.safe_load(f)
                            daq_backend = cfg.get('daq', {}).get('backend', 'unknown')
                            raman_provider = cfg.get('raman', {}).get('provider', 'unknown')
                            desc = f"({daq_backend} + {raman_provider})"
                    except:
                        pass
                typer.echo(f"  • {d.name} {desc}")
    
    # List profiles
    profiles_dir = Path("config/hardware_profiles")
    if profiles_dir.exists():
        typer.echo("\n🔧 Available Hardware Profiles:")
        for p in profiles_dir.glob("*.yaml"):
            typer.echo(f"  • {p.stem}")

@system_app.command("init")
def system_init(
    admin_email: str = typer.Option("admin@local", help="Admin email"),
    admin_name: str = typer.Option("Admin", help="Admin name")
):
    """Initialize the system database and create admin user"""
    typer.echo("🚀 Initializing system...")
    cmd = f'python -m retrofitkit.cli init --admin-email "{admin_email}" --admin-name "{admin_name}" --set-admin-password'
    return os.system(cmd)

@system_app.command("status") 
def system_status():
    """Show system status and configuration"""
    typer.echo("📊 System Status:")
    
    # Check config file
    config_file = Path("config/config.yaml")
    if config_file.exists():
        import yaml
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        
        system_cfg = cfg.get('system', {})
        daq_cfg = cfg.get('daq', {})
        raman_cfg = cfg.get('raman', {})
        
        typer.echo(f"  • Mode: {system_cfg.get('mode', 'unknown')}")
        typer.echo(f"  • DAQ Backend: {daq_cfg.get('backend', 'unknown')}")
        typer.echo(f"  • Raman Provider: {raman_cfg.get('provider', 'unknown')}")
        
        if daq_cfg.get('backend') == 'ni':
            ni_cfg = daq_cfg.get('ni', {})
            typer.echo(f"  • NI Device: {ni_cfg.get('device_name', 'unknown')}")
    else:
        typer.echo("  ❌ Configuration file not found")
    
    # Check database
    db_file = Path("data/audit.db")
    if db_file.exists():
        typer.echo(f"  • Database: {db_file} ({db_file.stat().st_size} bytes)")
    else:
        typer.echo("  • Database: Not initialized")

@system_app.command("logs")
def system_logs(
    follow: bool = typer.Option(False, "-f", help="Follow log output"),
    lines: int = typer.Option(50, "-n", help="Number of lines to show")
):
    """Show system logs"""
    log_file = Path("logs/app.log")
    if not log_file.exists():
        typer.echo("❌ Log file not found")
        return
    
    if follow:
        cmd = f"tail -f -n {lines} {log_file}"
    else:
        cmd = f"tail -n {lines} {log_file}"
    
    os.system(cmd)

@app.command()
def quickstart():
    """Interactive quickstart wizard"""
    typer.echo("🚀 Polymorph-4 Quickstart Wizard")
    typer.echo("=" * 40)
    
    # Check if system is initialized
    if not Path("config/config.yaml").exists():
        typer.echo("❌ System not found. Running installation...")
        run_script("install.py", "--full-setup")
        return
    
    typer.echo("✅ System detected")
    
    # Check hardware
    if typer.confirm("Configure hardware automatically?"):
        run_script("scripts/hardware_wizard.py")
    
    # Apply overlay
    if typer.confirm("Apply a configuration overlay?"):
        overlays = [d.name for d in Path("config/overlays").iterdir() if d.is_dir()]
        if overlays:
            typer.echo("Available overlays:")
            for i, overlay in enumerate(overlays):
                typer.echo(f"  [{i}] {overlay}")
            
            choice = typer.prompt("Select overlay number", type=int, default=0)
            if 0 <= choice < len(overlays):
                run_script("scripts/apply_overlay.py", overlays[choice], ".")
    
    # Start server
    if typer.confirm("Start the server?"):
        server()

if __name__ == "__main__":
    app()