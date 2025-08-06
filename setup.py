from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="micro-lm",
    version="0.1.0",
    author="microLM Contributors",
    description="A compact Transformer language model framework for learning and experimentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/microLM",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": [
            # GPU-accelerated PyTorch (install with: pip install -e .[gpu])
            # Note: This will install CPU version by default
            # For GPU support, run after installation:
            # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "all": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "microlm-train=src.training:main",
        ],
    },
    keywords="transformer, language model, pytorch, nlp, gpt, education",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/microLM/issues",
        "Source": "https://github.com/yourusername/microLM",
        "Documentation": "https://github.com/yourusername/microLM#readme",
    },
)
