# Polymorph-4 Quickstart Guide

This guide will get you up and running with Polymorph-4 in minutes.

## Prerequisites

- Python 3.11 or later
- pip (Python package manager)
- (Optional) Hardware: NI DAQ devices, Raman spectrometers

## Step 1: Choose Your Setup Path

### Path A: Complete Interactive Setup (Recommended)
```bash
python install.py --full-setup
```
This will:
1. Install all dependencies (including hardware drivers)
2. Initialize the database and create admin user
3. Guide you through hardware configuration
4. Let you choose a pre-built configuration overlay
5. Start the development server

### Path B: Quick Start with CLI
```bash
# Install system
python scripts/unified_cli.py install --hardware

# Run quickstart wizard  
python scripts/unified_cli.py quickstart
```

### Path C: Manual Step-by-Step
```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -r requirements-hw.txt  # Optional: hardware drivers

# 2. Initialize system
python -m retrofitkit.cli init --admin-email your@email.com --admin-name "Your Name" --set-admin-password

# 3. Configure hardware (choose one):
python scripts/hardware_wizard.py                    # Auto-detect
python scripts/apply_overlay.py NI_USB6343_Ocean0 .  # Use preset

# 4. Start server
uvicorn retrofitkit.api.server:app --host 0.0.0.0 --port 8000 --reload
```

## Step 2: Access the System

Open your web browser and go to:
- **Main Dashboard**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

Default login credentials will be created during initialization.

## Step 3: Verify Hardware Connection

1. Go to the hardware section in the web interface
2. Check that your devices are detected and configured
3. Run a test measurement to verify connectivity

## Step 4: Create Your First Recipe

1. Navigate to the recipes section
2. Create a new recipe or modify an existing example
3. Define your process steps and gating conditions
4. Save and execute your recipe

## Common Configurations

### NI USB-6343 + Ocean Optics
```bash
python scripts/unified_cli.py config overlay NI_USB6343_Ocean0
```

### NI PCIe-6363 + Horiba  
```bash
python scripts/unified_cli.py config overlay NI_PCIE6363_Horiba
```

### Red Pitaya + Andor
```bash
python scripts/unified_cli.py config overlay RedPitaya_Andor
```

### Simulation Only
```bash
python scripts/unified_cli.py config overlay NI_USB6343_Simulator
```

## Troubleshooting

### Hardware Not Detected
```bash
# Check what hardware is available
python scripts/unified_cli.py hardware list

# Re-run hardware wizard
python scripts/unified_cli.py hardware wizard
```

### Server Won't Start
```bash
# Check system status
python scripts/unified_cli.py system status

# View logs for errors
python scripts/unified_cli.py system logs
```

### Permission Issues (Linux/Mac)
```bash
# Make scripts executable
chmod +x install.py
chmod +x scripts/*.py
```

## Next Steps

- **Review Documentation**: Check `docs/` folder for detailed guides
- **Explore Recipes**: Look at examples in `recipes/` folder  
- **Configure Safety**: Set up E-stop and door interlocks
- **Set Up Monitoring**: Enable Prometheus + Grafana observability
- **Production Deployment**: Use Docker for production deployments

## Getting Help

1. Use the CLI help system: `python scripts/unified_cli.py --help`
2. Check system status: `python scripts/unified_cli.py system status`
3. Review logs: `python scripts/unified_cli.py system logs`
4. Read the full documentation in the `docs/` directory