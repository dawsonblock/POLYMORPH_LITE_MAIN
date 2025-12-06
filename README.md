# POLYMORPH-LITE

**Next-Generation Laboratory Automation Platform**

POLYMORPH-LITE is an open-source, modular platform for automating pharmaceutical and laboratory workflows. It integrates real-time data acquisition, AI-powered analysis, and 21 CFR Part 11 compliant audit logging into a unified system.

---

## ✨ Features

-   **Asynchronous Backend**: High-performance FastAPI backend with SQLAlchemy 2.0 async/await patterns.
-   **Visual Workflow Builder**: Design, validate, and execute complex automation protocols.
-   **Device Integration**: Unified driver model for spectrophotometers, DAQs, balances, and more.
-   **AI-Powered Analysis**: BentoML-integrated inference service for real-time spectral analysis.
-   **21 CFR Part 11 Compliance**: Immutable, tamper-evident audit logs with cryptographic hashing.
-   **Modern UI**: Next.js 15 frontend with Tailwind CSS and Shadcn/UI components.

---

## 🚀 Quick Start

### Prerequisites

-   Docker & Docker Compose
-   Node.js 18+ (for local frontend development)
-   Python 3.11+ (for local backend development)

### Run with Docker

```bash
# Clone the repository
git clone https://github.com/dawsonblock/POLYMORPH_LITE_MAIN.git
cd POLYMORPH_LITE_MAIN

# Build and start all services
make build
make up

# View logs
make logs
```

Access the application at:
-   **Frontend**: http://localhost:3001
-   **Backend API**: http://localhost:8001/api
-   **API Docs**: http://localhost:8001/docs

### Local Development

```bash
# Install dependencies
make install

# Run frontend
cd ui && npm run dev

# Run backend (in separate terminal)
source .venv/bin/activate
uvicorn main:app --reload --port 8001
```

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Next.js UI    │────▶│   FastAPI       │────▶│   PostgreSQL    │
│   (Port 3001)   │     │   (Port 8001)   │     │   (Port 5432)   │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   BentoML AI    │
                        │   (Port 3000)   │
                        └─────────────────┘
```

---

## 📁 Project Structure

```
├── retrofitkit/           # Core backend application
│   ├── api/               # FastAPI routers (endpoints)
│   ├── core/              # Business logic & workflows
│   ├── db/                # SQLAlchemy models & session
│   └── drivers/           # Hardware device drivers
├── ui/                    # Next.js frontend
├── bentoml_service/       # AI inference microservice
├── docker/                # Docker configurations
├── alembic/               # Database migrations
├── tests/                 # Pytest test suite
└── Makefile               # Developer commands
```

---

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
pytest --cov=retrofitkit tests/
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting a pull request.

---

**Built with ❤️ for the scientific community**