# Changelog

All notable changes to this project will be documented in this file.

## [v8.0.0] - 2025-11-29

### 🚀 Major Features
- **Hardware Integration**: Added drivers for Ocean Optics (Spectrometer) and Red Pitaya (DAQ).
- **Unified Pipeline**: Synchronized DAQ+Raman acquisition pipeline.
- **Workflow Engine**: Upgraded runner with Pause/Resume, Conditional Branching, and Human-in-the-loop support.
- **UI Overhaul**: New Next.js frontend with real-time Workflow Runner and Device Management.
- **AI Upgrade**: Versioned model training pipeline and batch inference service.

### 🛡️ Security & Compliance
- **Audit Logging**: 21 CFR Part 11 compliant tamper-proof audit trail.
- **RBAC**: Role-Based Access Control (Admin, Operator, Reviewer, Auditor).
- **Validation**: Automated IQ, OQ, and PQ validation suite.

### 🏗️ Infrastructure
- **Docker**: Full containerization of API, UI, and AI services.
- **CI/CD**: GitHub Actions pipeline for automated testing and validation.

### 🐛 Fixes
- Fixed workflow state persistence issues.
- Improved error handling in hardware drivers.
