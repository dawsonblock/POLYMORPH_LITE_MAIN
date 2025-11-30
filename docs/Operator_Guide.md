# POLYMORPH v8.0 Operator Guide

## Introduction
Welcome to POLYMORPH v8.0, the unified Lab Automation Platform. This guide covers daily operations, workflow execution, and troubleshooting.

## 1. Getting Started
### Login
Access the platform at `http://localhost:3000`.
- **Operator**: Use your assigned credentials.
- **Admin**: Contact IT for admin access.

### Dashboard Overview
- **Device Status**: Real-time health of connected instruments.
- **Active Workflows**: List of running or paused jobs.
- **Recent Alerts**: System warnings and errors.

## 2. Running a Workflow
1. Navigate to **Workflow Runner**.
2. Select a workflow (e.g., "Standard Acquisition").
3. Click **Start**.
4. Follow on-screen instructions for manual steps.
5. **Pause/Resume**: Use the controls if you need to intervene.

## 3. Device Management
- **Ocean Optics**: Ensure USB connection. Run "Self-Test" daily.
- **Red Pitaya**: Verify network connection (IP: 192.168.1.100).

## 4. Audit Trail
All actions are logged.
- View logs in the **Audit Log** page.
- Export logs to CSV for compliance reviews.

## 5. Troubleshooting
- **Device Offline**: Check cables and power. Restart the service via Docker.
- **Workflow Error**: Check the error message. Use "Retry" if applicable.
- **Support**: Contact support@polymorph.lab
