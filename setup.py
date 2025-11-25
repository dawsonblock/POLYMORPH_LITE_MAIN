"""
Setup configuration for POLYMORPH-4 Lite
Professional Python package installation
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    """Read requirements from file, filtering comments and empty lines."""
    with open(filename, 'r') as f:
        return [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]

# Core requirements
install_requires = read_requirements('requirements.txt')

# Optional hardware requirements
try:
    extras_require = {
        'hardware': read_requirements('requirements-hw.txt'),
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0', 
            'pytest-asyncio>=0.21.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
            'pre-commit>=3.0.0'
        ],
        'docs': [
            'mkdocs>=1.5.0',
            'mkdocs-material>=9.0.0',
            'mkdocstrings[python]>=0.20.0'
        ]
    }
except FileNotFoundError:
    extras_require = {
        'hardware': [],
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pytest-asyncio>=0.21.0', 
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
            'pre-commit>=3.0.0'
        ],
        'docs': [
            'mkdocs>=1.5.0',
            'mkdocs-material>=9.0.0',
            'mkdocstrings[python]>=0.20.0'
        ]
    }

# Add 'all' option that includes everything
extras_require['all'] = list(set(sum(extras_require.values(), [])))

setup(
    name="polymorph4-lite",
    version="1.0.0",
    author="POLYMORPH-4 Team",
    author_email="support@polymorph4.com",
    description="Unified Retrofit Control + Raman-Gating Kit for Analytical Instrument Automation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dawsonblock/POLYMORPH_Lite",
    project_urls={
        "Bug Tracker": "https://github.com/dawsonblock/POLYMORPH_Lite/issues",
        "Documentation": "https://github.com/dawsonblock/POLYMORPH_Lite/tree/main/docs",
        "Source Code": "https://github.com/dawsonblock/POLYMORPH_Lite",
        "Changelog": "https://github.com/dawsonblock/POLYMORPH_Lite/blob/main/CHANGELOG.md",
    },
    classifiers=[
        # Development Status
        "Development Status :: 5 - Production/Stable",
        
        # Intended Audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing", 
        "Intended Audience :: Healthcare Industry",
        
        # Topic
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: System :: Hardware :: Hardware Drivers",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        
        # Framework
        "Framework :: FastAPI",
        "Framework :: Pydantic",
    ],
    keywords=[
        "raman", "spectroscopy", "automation", "analytical", "instruments",
        "daq", "national-instruments", "ocean-optics", "horiba", "andor",
        "21cfr11", "compliance", "audit", "gmp", "pharmaceutical",
        "process-control", "crystallization", "monitoring"
    ],
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    python_requires=">=3.11",
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Package data
    include_package_data=True,
    package_data={
        "retrofitkit": [
            "api/static/*",
            "*.yaml",
            "*.json"
        ]
    },
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "polymorph4=retrofitkit.cli:main",
            "p4-cli=scripts.unified_cli:main",
            "p4-install=install:main",
            "p4-wizard=scripts.hardware_wizard:main",
        ]
    },
    
    # Additional data files
    data_files=[
        ("config", ["config/config.yaml", "config/raman.yaml"]),
        ("recipes", ["recipes/demo_gate.yaml"]),
        ("docs", ["README.md", "CHANGELOG.md", "CONTRIBUTING.md"]),
    ],
    
    # Zip safe
    zip_safe=False,
    
    # Minimum requirements check
    python_requires=">=3.11",
    
    # Development requirements
    test_suite="tests",
    
    # Project metadata
    platforms=["any"],
    license="MIT",
    
    # Additional metadata
    maintainer="POLYMORPH-4 Team",
    maintainer_email="support@polymorph4.com",
    
    # Options
    options={
        "bdist_wheel": {
            "universal": False,  # Not universal because we require Python 3.11+
        }
    }
)