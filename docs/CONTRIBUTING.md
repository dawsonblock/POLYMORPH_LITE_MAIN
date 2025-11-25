# Contributing to POLYMORPH-4 Lite

Thank you for your interest in contributing to POLYMORPH-4 Lite! We welcome contributions from the scientific and software development communities.

## 🎯 How to Contribute

### 🐛 Reporting Bugs
- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include system status output: `python scripts/unified_cli.py system status`
- Provide relevant log entries: `python scripts/unified_cli.py system logs`
- Include hardware configuration details

### ✨ Suggesting Features
- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Describe the scientific use case
- Consider hardware compatibility requirements
- Provide mockups or examples if applicable

### 🔧 Hardware Support
- Use the [hardware support template](.github/ISSUE_TEMPLATE/hardware_support.md)
- Include vendor documentation links
- Provide SDK/driver information
- Test with hardware wizard: `python scripts/unified_cli.py hardware wizard`

## 🛠️ Development Setup

### Prerequisites
- Python 3.11 or later
- Git
- Optional: Hardware devices for testing

### Setup Instructions
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/POLYMORPH_Lite.git
cd POLYMORPH_Lite

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy pre-commit

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
python -m pytest tests/
```

## 📝 Code Guidelines

### Code Style
- **Python**: Follow PEP 8, enforced by `black` and `flake8`
- **Type Hints**: Use type annotations (checked with `mypy`)
- **Docstrings**: Use Google-style docstrings
- **Line Length**: Maximum 127 characters

### Formatting
```bash
# Format code
black .

# Check linting
flake8 .

# Type checking
mypy retrofitkit --ignore-missing-imports
```

### Testing
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=retrofitkit

# Run specific test file
python -m pytest tests/test_recipe.py
```

## 🏗️ Project Structure

```
POLYMORPH_Lite/
├── retrofitkit/           # Core application
│   ├── api/              # FastAPI server & routes
│   ├── core/             # Recipe engine & orchestration
│   ├── drivers/          # Hardware drivers
│   ├── safety/           # Safety systems
│   ├── compliance/       # Audit & signatures
│   └── data/            # Database models
├── config/               # Configuration management
├── scripts/              # CLI tools & utilities
├── docs/                # Documentation
├── tests/               # Test suite
└── docker/              # Container configurations
```

## 🔄 Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
# or  
git checkout -b hardware/vendor-model
```

### 2. Implement Changes
- Write clean, well-documented code
- Add tests for new functionality
- Update documentation as needed
- Follow existing patterns and conventions

### 3. Test Your Changes
```bash
# Run tests
python -m pytest

# Test with real hardware (if available)
python scripts/unified_cli.py hardware list
python scripts/unified_cli.py quickstart

# Test Docker build
docker build -f Dockerfile.multi --target development .
```

### 4. Commit Changes
```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat(drivers): add support for Vendor Model X

- Implement VendorX driver class
- Add configuration validation  
- Include integration tests
- Update documentation

Closes #123"
```

### 5. Submit Pull Request
- Push your branch to your fork
- Create pull request against `main` branch
- Fill out the PR template completely
- Link related issues
- Ensure CI tests pass

## 🧪 Testing Guidelines

### Test Categories
1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **Hardware Tests**: Test with real devices (when available)
4. **End-to-End Tests**: Test complete workflows

### Writing Tests
```python
import pytest
from retrofitkit.core.recipe import RecipeEngine

def test_recipe_validation():
    """Test recipe YAML validation."""
    recipe_yaml = """
    name: Test Recipe
    steps:
      - type: bias_set
        voltage: 2.5
    """
    engine = RecipeEngine()
    recipe = engine.parse_recipe(recipe_yaml)
    assert recipe.name == "Test Recipe"
    assert len(recipe.steps) == 1

@pytest.mark.hardware
def test_ni_daq_connection():
    """Test NI DAQ hardware connection (requires hardware)."""
    # This test only runs when hardware is available
    pass
```

### Running Hardware Tests
```bash
# Skip hardware tests (default)
python -m pytest

# Include hardware tests (requires devices)
python -m pytest -m hardware

# Run specific hardware vendor tests
python -m pytest -k "ni_daq"
```

## 📚 Documentation

### Adding Documentation
- Update relevant `.md` files in `docs/`
- Include code examples and usage patterns
- Document configuration options
- Add troubleshooting information

### Documentation Standards
- Use clear, concise language
- Include practical examples
- Provide troubleshooting steps
- Link to related sections
- Keep hardware compatibility matrices updated

## 🔧 Hardware Driver Development

### Adding New Hardware Support

1. **Create Driver Class**
```python
# retrofitkit/drivers/raman/vendor_newvendor.py
from retrofitkit.drivers.raman.base import RamanBase

class NewVendorRaman(RamanBase):
    def __init__(self, config):
        # Initialize vendor SDK
        pass
        
    async def read_frame(self):
        # Implement data acquisition
        pass
```

2. **Add to Factory**
```python
# retrofitkit/drivers/raman/factory.py
def create_raman_driver(provider, config):
    if provider == "newvendor":
        from .vendor_newvendor import NewVendorRaman
        return NewVendorRaman(config)
```

3. **Create Configuration Overlay**
```yaml
# config/overlays/NI_USB6343_NewVendor/config.yaml
system:
  mode: production
raman:
  provider: newvendor
  vendor:
    sdk_path: /path/to/sdk
```

4. **Add Tests**
```python
# tests/test_newvendor_driver.py
def test_newvendor_initialization():
    """Test NewVendor driver initialization."""
    pass
```

5. **Update Documentation**
- Add to hardware support matrix
- Document configuration options
- Include setup instructions
- Add troubleshooting guide

## 🚀 Release Process

### Version Numbering
- **Major**: Breaking changes (v2.0.0)
- **Minor**: New features (v1.1.0)  
- **Patch**: Bug fixes (v1.0.1)

### Release Checklist
- [ ] Update version in `VERSION.md`
- [ ] Update `CHANGELOG.md`
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Test Docker builds
- [ ] Create release notes
- [ ] Tag release: `git tag v1.1.0`

## 💬 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers get started
- Share knowledge and resources
- Report inappropriate behavior

### Communication Channels
- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Requests**: Code contributions
- **Documentation**: In-code and markdown docs

## 🏆 Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- GitHub contributor graphs
- Special recognition for major contributions

### Types of Contributions
- 🐛 **Bug fixes**
- ✨ **New features** 
- 🔧 **Hardware support**
- 📚 **Documentation**
- 🧪 **Testing**
- 🎨 **UI/UX improvements**
- 🚀 **Performance optimizations**

Thank you for contributing to POLYMORPH-4 Lite and advancing scientific automation! 🔬