"""Setup configuration for Wagehood Trading System"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="wagehood",
    version="0.1.0",
    author="Wagehood Team",
    description="Professional trend-following trading analysis system with Redis-based worker service and CLI tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=['src', 'src.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    entry_points={
        "console_scripts": [
            "wagehood-cli=market_analysis_cli:cli",
            "wagehood-watch=market_watch:main",
            "wagehood-realtime=run_realtime:main",
        ],
    },
)