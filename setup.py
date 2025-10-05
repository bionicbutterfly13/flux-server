"""
Flux Server - Consciousness-Enhanced Knowledge Retrieval
Setup configuration for pip installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="flux-server",
    version="1.0.0",
    author="bionicbutterfly13",
    description="Consciousness-enhanced knowledge retrieval microservice",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bionicbutterfly13/flux-server",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "python-multipart==0.0.6",
        "pydantic==2.5.0",
        "pydantic-settings==2.1.0",
        "neo4j==5.14.0",
        "redis==5.0.1",
        "httpx==0.25.1",
        "numpy==1.26.0",
        "uvloop==0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest==7.4.3",
            "pytest-asyncio==0.21.1",
        ]
    },
    entry_points={
        "console_scripts": [
            "flux-server=src.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
)
