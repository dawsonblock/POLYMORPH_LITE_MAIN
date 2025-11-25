# POLYMORPH-4 Lite User Manual

**Version**: 2.0.0  
**Last Updated**: November 25, 2025

---

## Table of Contents
1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Getting Started](#getting-started)
4. [Dashboard Guide](#dashboard-guide)
5. [Running Experiments](#running-experiments)
6. [Monitoring & Alerts](#monitoring--alerts)
7. [Data Management](#data-management)
8. [Troubleshooting](#troubleshooting)
9. [Safety & Compliance](#safety--compliance)

---

## Introduction

POLYMORPH-4 Lite is an AI-powered laboratory automation platform for real-time polymorph detection and crystallization monitoring. The system integrates:

- **Raman Spectroscopy**: Real-time molecular analysis
- **DAQ (Data Acquisition)**: Environmental sensors (temperature, pressure, flow)
- **AI Analysis**: Deep learning-based polymorph classification
- **Automated Control**: Intelligent experiment management

### Key Features
- ✅ Real-time spectral analysis with AI inference
- ✅ Automated crystallization detection
- ✅ 21 CFR Part 11 compliant audit trails
- ✅ Electronic signatures with PKI
- ✅ Role-based access control (RBAC)
- ✅ Multi-factor authentication (MFA)
- ✅ Crash recovery and state persistence

---

## System Overview

### Architecture
```
┌─────────────────┐
│   Web Browser   │ ← Users access via modern web UI
└────────┬────────┘
         │ HTTPS
┌────────▼────────┐
│  Frontend (GUI) │ ← React + Vite + TailwindCSS
└────────┬────────┘
         │ WebSocket + REST API
┌────────▼────────┐
│ Backend (API)   │ ← FastAPI + Python 3.11
└────────┬────────┘
         │
    ┌────┴────┬───────────┬──────────┐
    │         │           │          │
┌───▼───┐ ┌──▼──┐  ┌─────▼─────┐ ┌─▼──┐
│ Raman │ │ DAQ │  │ AI Service│ │ DB │
│ Driver│ │     │  │ (BentoML) │ │    │
└───────┘ └─────┘  └───────────┘ └────┘
```

### Hardware Components
- **Raman Spectrometer**: Horiba LabRAM or compatible
- **DAQ Module**: Gamry potentiostat or NI-DAQ
- **Pumps & Sensors**: Temperature, pressure, flow rate monitoring

---

## Getting Started

### First Time Login

1. **Access the System**
   - Open your web browser
   - Navigate to: `http://localhost:3000` (or your deployment URL)

2. **Login Credentials**
   - Default admin account: `admin@polymorph.lab`
   - Password: (provided by system administrator)
   - **Important**: Change your password on first login

3. **Enable MFA (Recommended)**
   - Go to Settings → Security
   - Click "Enable Multi-Factor Authentication"
   - Scan QR code with authenticator app (Google Authenticator, Authy, etc.)
   - Enter verification code

### User Roles

| Role | Permissions |
|------|-------------|
| **Admin** | Full system access, user management, configuration |
| **Operator** | Run experiments, view data, create audit logs |
| **Analyst** | View-only access to data and reports |
| **Guest** | Limited dashboard view only |

---

## Dashboard Guide

### Main Dashboard

The dashboard provides a real-time overview of system status:

#### Status Cards (Top Row)
- **System Status**: Overall health indicator
  - 🟢 Green = Healthy
  - 🟡 Yellow = Warning
  - 🔴 Red = Error
- **Temperature**: Current reactor temperature with progress bar
- **Pressure**: System pressure in bar
- **Flow Rate**: mL/min flow rate

#### Real-time Spectrum View
- Displays live Raman spectral data
- X-axis: Wavenumber (cm⁻¹)
- Y-axis: Intensity (arbitrary units)
- Updates every 500ms during active experiments

#### Active Processes
- Shows currently running experiments
- Progress bar indicates completion percentage
- Step counter shows current/total steps

#### Recent Alerts
- System notifications and warnings
- Color-coded by severity:
  - 🔴 Error: Immediate attention required
  - 🟡 Warning: Monitor closely
  - 🔵 Info: Status updates
  - 🟢 Success: Task completed

#### Component Health
- Real-time status of all subsystems
- Glowing green dot = Online and healthy
- Click component for detailed diagnostics

---

## Running Experiments

### Creating a New Recipe

1. **Navigate to Recipes**
   - Click "Recipes" in the sidebar
   - Click "New Recipe" button

2. **Recipe Configuration**
   ```yaml
   Name: Aspirin Crystallization Study
   Description: Form I → Form II transformation
   Duration: 120 minutes
   Temperature Range: 20-80°C
   Flow Rate: 2.0 mL/min
   ```

3. **Add Steps**
   - **Step 1**: Heat to 60°C
   - **Step 2**: Hold for 30 min
   - **Step 3**: Cool to 25°C at 1°C/min
   - **Step 4**: Monitor for 60 min

4. **Enable AI Monitoring**
   - ✅ Real-time polymorph detection
   - ✅ Auto-alert on phase transition
   - ✅ Emergency stop on unexpected crystallization

### Starting an Experiment

1. **Pre-Flight Checks**
   - Verify all components show "Healthy" status
   - Ensure sample is loaded in reactor
   - Check solvent levels
   - Confirm temperature calibration

2. **Start Procedure**
   - Select recipe from dropdown
   - Review parameters
   - Click "Start Experiment"
   - **Sign with Electronic Signature**:
     - Enter your username
     - Enter password
     - Provide reason for execution
     - Click "Sign & Execute"

3. **During Experiment**
   - Monitor real-time dashboard
   - Watch for AI alerts
   - Do not interrupt unless emergency
   - All actions are audit-logged

### Emergency Stop

⚠️ **Use ONLY in emergencies**

1. Click red "EMERGENCY STOP" button
2. Provide reason in dialog
3. Sign with electronic signature
4. System will:
   - Halt all pumps
   - Save current state
   - Log emergency stop event
   - Trigger safety protocols

---

## Monitoring & Alerts

### Alert Types

| Alert | Description | Action Required |
|-------|-------------|-----------------|
| **New Polymorph Detected** | AI identified phase transition | Review spectral data, log observation |
| **Temperature Deviation** | ±2°C from setpoint | Check cooling system |
| **Pressure Anomaly** | Outside safe range | Inspect valves and connections |
| **Hardware Disconnected** | Lost connection to device | Check cables, restart driver |
| **AI Service Unavailable** | Inference service down | Contact administrator |

### Email Notifications

Configure email alerts:
1. Settings → Notifications
2. Enter email address
3. Select alert types to receive
4. Test notification

---

## Data Management

### Accessing Experiment Data

1. **Navigate to Data Explorer**
   - Sidebar → Data
   - Filter by date, user, or experiment ID

2. **Export Data**
   - Select experiment
   - Click "Export"
   - Choose format:
     - CSV (spectral data)
     - PDF (report with charts)
     - JSON (raw data)

3. **Data Retention**
   - Raw data: 2 years
   - Processed results: 5 years
   - Audit logs: 7 years (regulatory requirement)

### Audit Trail

All actions are logged per 21 CFR Part 11:
- User login/logout
- Experiment start/stop
- Parameter changes
- Data exports
- System configuration changes

**Audit Log Fields:**
- Timestamp (UTC)
- User ID
- Action type
- IP address
- Electronic signature (SHA-256)
- Reason for change

---

## Troubleshooting

### Common Issues

#### Issue: "Raman Driver Not Responding"
**Solution:**
1. Check USB connection to spectrometer
2. Restart Raman service: `systemctl restart raman-driver`
3. Verify spectrometer power
4. Check driver logs: `/var/log/polymorph/raman.log`

#### Issue: "AI Service Connection Failed"
**Solution:**
1. Verify BentoML service is running:
   ```bash
   docker ps | grep polymorph
   ```
2. Restart AI service:
   ```bash
   docker restart polymorph-ai
   ```
3. Check circuit breaker status in System Monitor

#### Issue: "Temperature Reading -999°C"
**Solution:**
- Sensor disconnected or faulty
- Check thermocouple connection
- Calibrate sensor: Settings → Hardware → Temperature Calibration

#### Issue: "Experiment Won't Start"
**Possible Causes:**
- ❌ Another experiment already running
- ❌ Hardware not ready
- ❌ Insufficient user permissions
- ❌ Invalid recipe parameters

### Support Contacts

- **Technical Support**: support@polymorph.lab
- **Emergency Hotline**: +1-555-POLY-911
- **Documentation**: https://docs.polymorph.lab

---

## Safety & Compliance

### Laboratory Safety

⚠️ **Always wear appropriate PPE:**
- Safety glasses
- Lab coat
- Chemical-resistant gloves

⚠️ **Chemical Hazards:**
- Refer to SDS (Safety Data Sheet) for all chemicals
- Use fume hood for volatile solvents
- Dispose of waste per institutional policy

### Regulatory Compliance

**21 CFR Part 11 Requirements:**
- ✅ Secure, computer-generated, time-stamped audit trails
- ✅ Electronic signatures equivalent to handwritten signatures
- ✅ System validation and documentation
- ✅ Limited system access to authorized users
- ✅ Use of operational system checks
- ✅ Authority checks to ensure authorized use
- ✅ Device checks to determine validity of data source
- ✅ Education, training, and experience requirements

**Data Integrity (ALCOA+ Principles):**
- **Attributable**: All records linked to responsible individual
- **Legible**: Data is readable and permanent
- **Contemporaneous**: Recorded at time of activity
- **Original**: First capture of data
- **Accurate**: Free from errors, verified
- **Complete**: All data available
- **Consistent**: Timestamps in expected sequence
- **Enduring**: Durable and backed up
- **Available**: Readily accessible for review

---

## Appendix

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + N` | New Recipe |
| `Ctrl + S` | Save Current State |
| `Ctrl + E` | Export Data |
| `Ctrl + M` | Open System Monitor |
| `F5` | Refresh Dashboard |
| `Esc` | Cancel Current Action |

### Glossary

- **Polymorph**: Different crystalline forms of the same compound
- **Raman Spectroscopy**: Vibrational spectroscopy technique
- **DAQ**: Data Acquisition System
- **BentoML**: ML model serving framework
- **Circuit Breaker**: Fault tolerance pattern

---

**© 2025 POLYMORPH-4 Research Team. All Rights Reserved.**
