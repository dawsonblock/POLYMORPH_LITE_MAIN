# POLYMORPH-4 Lite Testing Guide

## Overview

This document describes the comprehensive testing strategy for POLYMORPH-4 Lite, including unit tests, integration tests, E2E tests, and performance tests.

## Test Coverage Summary

| Component | Coverage | Test Files | Test Count |
|-----------|----------|------------|------------|
| Backend (Python) | ~35-40% | 14 files | 300+ tests |
| Frontend (TypeScript) | ~15-20% | 3 files | 50+ tests |
| E2E (Playwright) | Critical paths | 3 files | 25+ tests |
| Integration | Key workflows | 1 file | 20+ tests |
| Performance | Benchmarks | 1 file | 15+ tests |

## Test Types

### 1. Unit Tests

#### Backend (Python/pytest)

**Location:** `tests/`

**Test Files:**
- `test_api_auth.py` - Authentication endpoints
- `test_api_security.py` - Security middleware and JWT validation
- `test_security_validators.py` - Input validation and sanitization
- `test_compliance_signatures.py` - Electronic signatures (21 CFR Part 11)
- `test_compliance_approvals.py` - Multi-role approval workflows
- `test_safety_interlocks.py` - Safety interlock systems
- `test_safety_watchdog.py` - Watchdog monitoring
- `test_workflows_engine.py` - Workflow parsing and execution
- `test_api_devices.py` - Device management APIs
- `test_api_workflows.py` - Workflow APIs
- `test_database.py` - Database operations

**Running Backend Tests:**

```bash
# All unit tests (excluding integration/performance)
pytest tests/ -v -m "not integration and not performance and not slow"

# With coverage
pytest tests/ --cov=retrofitkit --cov-report=html --cov-report=term-missing

# Specific test file
pytest tests/test_api_auth.py -v

# Specific test class
pytest tests/test_api_auth.py::TestLoginEndpoint -v

# Specific test
pytest tests/test_api_auth.py::TestLoginEndpoint::test_successful_login -v
```

#### Frontend (TypeScript/Vitest)

**Location:** `gui-v2/frontend/src/**/__tests__/`

**Test Files:**
- `pages/__tests__/login.test.tsx` - Login page component
- `components/__tests__/SpectralView.test.tsx` - Chart component
- `stores/__tests__/auth-store.test.ts` - Authentication store

**Running Frontend Tests:**

```bash
cd gui-v2/frontend

# All tests
npm test

# Watch mode
npm test -- --watch

# Coverage
npm test -- --coverage

# Specific file
npm test -- login.test.tsx
```

### 2. Integration Tests

**Location:** `tests/test_integration_workflows.py`

**Test Scenarios:**
- Complete workflow execution (auth → upload → execute → verify)
- Multi-user approval workflows
- Concurrent workflow execution
- Device and workflow integration
- Audit trail through workflow
- E-signature with approvals

**Running Integration Tests:**

```bash
# All integration tests
pytest tests/ -v -m "integration"

# With backend services running
pytest tests/test_integration_workflows.py -v
```

**Requirements:**
- Backend API server running
- Test database
- Mock devices or simulators

### 3. End-to-End (E2E) Tests

**Location:** `gui-v2/frontend/e2e/`

**Framework:** Playwright

**Test Files:**
- `login.spec.ts` - Login flow and session management
- `dashboard.spec.ts` - Dashboard navigation and features
- `workflows.spec.ts` - Workflow management and data visualization

**Running E2E Tests:**

```bash
cd gui-v2/frontend

# Install Playwright browsers (first time only)
npx playwright install

# Run all E2E tests
npx playwright test

# Run specific browser
npx playwright test --project=chromium

# Run in headed mode (see browser)
npx playwright test --headed

# Run specific test file
npx playwright test e2e/login.spec.ts

# Debug mode
npx playwright test --debug
```

**E2E Test Reports:**

```bash
# View HTML report
npx playwright show-report

# Report location: gui-v2/frontend/playwright-report/
```

**Requirements:**
- Backend API running at `http://localhost:8000`
- Frontend dev server at `http://localhost:5173`
- Test users created in database

### 4. Performance Tests

**Location:** `tests/test_performance.py`

**Test Categories:**
- API response time benchmarks
- Workflow execution performance
- Database query performance
- Memory usage patterns
- Load and stress testing
- Throughput measurements

**Running Performance Tests:**

```bash
# All performance tests
pytest tests/ -v -m "performance"

# Benchmark tests only
pytest tests/test_performance.py -v -m "benchmark" --benchmark-only

# With benchmark comparison
pytest tests/test_performance.py --benchmark-compare

# Generate benchmark JSON
pytest tests/test_performance.py --benchmark-json=benchmark.json
```

**Performance Targets:**
- Login: < 500ms
- Workflow list: < 100ms
- Simple workflow execution: < 200ms
- Throughput: > 10 workflows/second
- Concurrent uploads: < 5s for 20 workflows

## Test Markers

Tests are categorized using pytest markers:

- `integration` - Integration tests requiring external services
- `slow` - Slow-running tests (> 10s)
- `performance` - Performance and benchmark tests
- `e2e` - End-to-end tests
- `hardware` - Tests requiring physical hardware

**Running by Marker:**

```bash
# Skip integration tests
pytest -m "not integration"

# Only integration tests
pytest -m "integration"

# Multiple markers
pytest -m "integration or performance"

# Exclude multiple
pytest -m "not integration and not slow"
```

## Coverage Requirements

### Coverage Thresholds

**Overall:**
- Backend: ≥ 30% (current target)
- Frontend: ≥ 15% (current target)

**Critical Modules:** (≥ 60% required)
- `retrofitkit/api/auth.py`
- `retrofitkit/api/security.py`
- `retrofitkit/compliance/*`
- `retrofitkit/safety/*`

**Production Ready:** (≥ 80% recommended)
- All security-critical modules
- Compliance and audit systems
- Safety interlock systems

### Generating Coverage Reports

**Backend:**

```bash
# Terminal report
pytest --cov=retrofitkit --cov-report=term-missing

# HTML report (opens in browser)
pytest --cov=retrofitkit --cov-report=html
open htmlcov/index.html

# XML report (for CI/CD)
pytest --cov=retrofitkit --cov-report=xml

# Combine with specific tests
pytest tests/test_api_*.py --cov=retrofitkit/api --cov-report=html
```

**Frontend:**

```bash
cd gui-v2/frontend

# Coverage report
npm test -- --coverage

# Open HTML report
open coverage/lcov-report/index.html
```

## CI/CD Integration

### GitHub Actions Workflows

**Main CI Pipeline:** `.github/workflows/ci-enhanced.yml`

**Stages:**
1. **Backend Tests** - Matrix testing (Python 3.11, 3.12)
   - Linting (ruff, mypy)
   - Unit tests with coverage
   - Integration tests
   - Coverage thresholds check

2. **Frontend Tests**
   - Linting (ESLint)
   - Type checking (TypeScript)
   - Unit tests with coverage

3. **Performance Tests** (main branch only)
   - Benchmark tests
   - Performance regression detection

4. **Security Scan**
   - Bandit security scan
   - Safety dependency check

5. **Docker Build**
   - Multi-stage build test
   - Image size optimization

6. **Coverage Summary**
   - Combined coverage report
   - PR comments with results

**Running Locally (like CI):**

```bash
# Backend CI steps
pip install ruff mypy pytest pytest-cov
ruff check retrofitkit/
mypy retrofitkit/ --ignore-missing-imports
pytest tests/ -m "not integration" --cov=retrofitkit --cov-fail-under=30

# Frontend CI steps
cd gui-v2/frontend
npm run lint
npx tsc --noEmit
npm test -- --coverage --run
```

## Test Data and Fixtures

### Backend Fixtures (`tests/conftest.py`)

- `temp_db_dir` - Temporary database directory
- `mock_config` - Mock application configuration
- `mock_daq` - Mock DAQ device
- `app_context` - Mock application context
- `test_users` - Pre-configured test users

### Frontend Mocks

**Store Mocks:**
- `useAuthStore` - Authentication store
- `useSystemStore` - System state store

**Component Mocks:**
- `framer-motion` - Animation library
- `recharts` - Chart components
- `sonner` - Toast notifications

## Best Practices

### Writing Tests

1. **Use Descriptive Names**
   ```python
   def test_login_with_invalid_credentials_returns_401():
       # Clear what is being tested
   ```

2. **Arrange-Act-Assert Pattern**
   ```python
   def test_create_user():
       # Arrange
       users = Users()

       # Act
       users.create("test@example.com", "Test User", "Operator", "pass")

       # Assert
       assert user exists
   ```

3. **Test One Thing**
   - Each test should verify one specific behavior
   - Split complex tests into multiple smaller tests

4. **Use Fixtures for Setup**
   - Don't repeat setup code
   - Use pytest fixtures or beforeEach

5. **Mock External Dependencies**
   - Database calls
   - API requests
   - Hardware interfaces

### Coverage Guidelines

1. **Prioritize Critical Code**
   - Security and authentication
   - Compliance and audit
   - Safety systems

2. **Test Edge Cases**
   - Null/empty inputs
   - Boundary conditions
   - Error conditions

3. **Don't Chase 100%**
   - Focus on meaningful tests
   - Some code doesn't need tests (simple getters, etc.)

## Troubleshooting

### Common Issues

**1. Module Import Errors**
```bash
# Ensure you're in project root
cd /path/to/POLYMORPH_LITE_MAIN

# Install all dependencies
pip install -r requirements.txt
pip install pytest pytest-cov pytest-asyncio
```

**2. Database Errors**
```bash
# Clear test databases
rm -rf /tmp/test_data_*

# Use temp directory in tests
os.environ["P4_DATA_DIR"] = tempfile.mkdtemp()
```

**3. Frontend Test Failures**
```bash
# Clear node modules and reinstall
cd gui-v2/frontend
rm -rf node_modules package-lock.json
npm install

# Update test library
npm install -D @testing-library/react@latest
```

**4. E2E Test Timeouts**
```bash
# Ensure services are running
# Backend: http://localhost:8000
# Frontend: http://localhost:5173

# Increase timeout in playwright.config.ts
timeout: 60000
```

## Continuous Improvement

### Future Enhancements

1. **Additional Test Coverage**
   - Data storage and retrieval tests
   - Metrics and observability tests
   - Remaining driver tests (Horiba, Andor, etc.)

2. **Advanced Testing**
   - Contract testing for APIs
   - Visual regression testing
   - Mutation testing

3. **Performance Optimization**
   - Load testing with realistic data
   - Database query optimization
   - Frontend rendering performance

4. **Test Infrastructure**
   - Test data factories
   - Snapshot testing
   - Parallel test execution

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Vitest Documentation](https://vitest.dev/)
- [Playwright Documentation](https://playwright.dev/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [21 CFR Part 11 Compliance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/part-11-electronic-records-electronic-signatures-scope-and-application)

## Support

For issues or questions about testing:
1. Check this documentation
2. Review existing tests for examples
3. Check CI/CD logs for detailed error messages
4. Consult the team or create an issue

---

**Last Updated:** 2024-11-26
**Test Suite Version:** 2.0
**Maintained by:** POLYMORPH-4 Development Team
