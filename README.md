# 🤖 microLM: A Compact Transformer Language Model Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **lightweight and easy-to-understand** framework for training and experimenting with Transformer-based language models.

## ✨ Features

- 🏗️ **Clean Architecture**: Modular design with clear separation of concerns
- ⚡ **Modern PyTorch**: Supports Flash Attention for PyTorch >= 2.0
- 📚 **Educational**: Comprehensive examples and documentation for learning
- 🔧 **Configurable**: Easy model configuration via dataclasses
- 📦 **Efficient Training**: Packed dataset implementation for optimal memory usage
- 🎯 **GPT-Style**: Implements causal self-attention and standard transformer blocks
- 🚀 **Ready-to-Use**: Pre-configured for TinyStories dataset

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (Python 3.11+ recommended)
- PyTorch 2.0+ (CPU or GPU version)

### Installation

#### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd microLM
```

#### Step 2: Install the Package

**Option A: Development Installation with GPU Support (Recommended)**

For the best performance with GPU acceleration:

```bash
# 1. Install the package in editable mode
pip install -e .

# 2. Install PyTorch with GPU support (CUDA)
# For CUDA 12.1 (most common):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (if you have older drivers):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Option B: CPU-Only Installation**

If you don't have a GPU or prefer CPU-only:

```bash
# Install with CPU-only PyTorch
pip install -e .
# PyTorch CPU version will be installed automatically
```

**Option C: Regular Installation**

```bash
pip install -r requirements.txt
# Then follow GPU installation steps above if desired
```

#### Step 3: Verify Installation

**Basic Installation Check:**
```bash
# Test the installation
python -c "import micro_lm; print('microLM installed successfully!')"
```

**GPU Support Verification:**
```bash
# Check if GPU is available and working
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

**Quick GPU Detection:**
```bash
# Check your GPU (Windows)
nvidia-smi

# Check your GPU (Linux)
lspci | grep -i nvidia
```

**Expected Output for GPU Users:**
```
PyTorch version: 2.5.1+cu124
CUDA available: True
GPU device: NVIDIA GeForce RTX 4060 Laptop GPU
```

### 🔧 Setting Up Python Alias (Windows)

If you have multiple Python versions or long paths, create an alias for easier usage:

**PowerShell (Temporary - current session only):**
```powershell
# Set alias for current session
Set-Alias -Name python -Value "C:\Users\YourUsername\AppData\Local\Programs\Python\Python311\python.exe"

# Now you can use:
python examples/model_inspection_demo.py
```

**PowerShell Profile (Permanent):**
```powershell
# Open/create your PowerShell profile
notepad $PROFILE

# Add this line to the file:
Set-Alias -Name python -Value "C:\Users\YourUsername\AppData\Local\Programs\Python\Python311\python.exe"

# Reload profile
. $PROFILE
```

**Alternative: Use `py` launcher (if available):**
```powershell
# Use Python launcher (automatically finds correct version)
py examples/model_inspection_demo.py
```

### 🔍 Troubleshooting

**Problem: `ModuleNotFoundError: No module named 'micro_lm'`**

*Solution:* The package isn't installed. Run the development installation:
```bash
pip install -e .
```

**Problem: Multiple Python versions causing conflicts**

*Solution:* Use the full path to your desired Python version:
```bash
# Find your Python installation
where python  # Windows
which python  # Linux/Mac

# Use specific version
C:\Path\To\Python311\python.exe -m pip install -e .
```

**Problem: `'python' is not recognized as an internal or external command`**

*Solution:* Either:
1. Add Python to your PATH environment variable, or
2. Use the full path to python.exe, or
3. Set up an alias as shown above

**Problem: Import errors with dependencies**

*Solution:* Install missing dependencies:
```bash
pip install torch datasets transformers tokenizers matplotlib wandb tqdm
```

### Basic Usage

```python
from micro_lm.model import MicroLM
from micro_lm.config import ModelConfig

# Create a small model
config = ModelConfig(
    block_size=256,
    vocab_size=1024,
    n_layer=6,
    n_head=6,
    n_embd=384
)

model = MicroLM(config)
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
```

## 📖 Examples

The `examples/` directory contains comprehensive scripts demonstrating all core functionalities: 

### 📊 Data Preparation (`data_preparation_demo.py`)

Learn the complete data pipeline:
- ✅ Initialize the tokenizer
- ✅ Load datasets from Hugging Face (TinyStories)
- ✅ Create packed datasets for efficient training
- ✅ Inspect batches and data flow

```bash
python examples/data_preparation_demo.py
```

### 🔍 Model Inspection (`model_inspection_demo.py`)

Explore model architecture and parameters:
- ✅ Configure model architecture
- ✅ Print detailed model summary
- ✅ Calculate parameter counts
- ✅ Understand model structure

```bash
python examples/model_inspection_demo.py
```

### 🏋️ Training Demo (`training_demo.py`)

Minimal training loop implementation:
- ✅ Model, tokenizer, and dataset setup
- ✅ AdamW optimizer configuration
- ✅ Forward pass and loss calculation
- ✅ Backpropagation and parameter updates

```bash
python examples/training_demo.py
```

### 💬 Chat Demo (`chat_demo.py`)

Interactive chat interface:
- ✅ Load pre-trained models
- ✅ Interactive conversation loop
- ✅ Text generation with sampling

```bash
python examples/chat_demo.py
```

### 📝 Text Generation (`text_generation_demo.py`)

Text generation capabilities:
- ✅ Load trained models
- ✅ Generate text with various sampling strategies
- ✅ Control generation parameters

```bash
python examples/text_generation_demo.py
```

## 🎯 Training

### Quick Training

Train a model from scratch on the TinyStories dataset:

```bash
python src/training.py
```

### Custom Training

```python
from micro_lm.config import get_config
from micro_lm.train import train_model

# Use pre-configured settings
config = get_config('tinystories')  # or 'base'

# Customize as needed
config.batch_size = 32
config.learning_rate = 1e-4
config.max_iters = 10000

# Start training
train_model(config)
```

## 🏗️ Architecture

The model implements a standard GPT-style transformer with:

- **Causal Self-Attention**: Multi-head attention with causal masking
- **Feed-Forward Networks**: MLP blocks with GELU activation
- **Layer Normalization**: Optional bias support
- **Positional Embeddings**: Learned position encodings
- **Weight Tying**: Shared embedding and output layer weights

## 📁 Project Structure

```
microLM/
├── micro_lm/           # Core framework
│   ├── model.py        # Transformer implementation
│   ├── config.py       # Configuration classes
│   ├── dataset.py      # Data loading utilities
│   ├── tokenizer.py    # Tokenization
│   ├── train.py        # Training utilities
│   └── inference.py    # Inference utilities
├── examples/           # Educational demos
├── src/               # Training scripts
├── tests/             # Test suite
└── requirements.txt   # Dependencies
```

## ⚙️ Configuration

Easily configure your models:

```python
from micro_lm.config import ModelConfig, TrainConfig

# Model architecture
model_config = ModelConfig(
    block_size=512,      # Context length
    vocab_size=50257,    # Vocabulary size
    n_layer=12,          # Number of layers
    n_head=12,           # Attention heads
    n_embd=768,          # Embedding dimension
    dropout=0.1          # Dropout rate
)

# Training configuration
train_config = TrainConfig(
    model_config=model_config,
    batch_size=64,
    learning_rate=1e-4,
    max_iters=50000
)
```

## Project Structure

```
. 
├── examples/           # Example scripts
├── micro_lm/           # Core library source code
│   ├── __init__.py
│   ├── config.py       # Configuration objects (ModelConfig, TrainConfig)
│   ├── dataset.py      # PackedDataset for streaming and batching
│   ├── inference.py    # Generator class for text generation
│   ├── model.py        # GPT model architecture (MicroLM)
│   ├── train.py        # Trainer class for the training loop
│   └── tokenizer.py    # MicroTokenizer for text tokenization
├── tests/              # Unit tests for all modules
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_inference.py
│   ├── test_model.py
│   └── test_tokenizer.py
├── README.md           # This file
└── requirements.txt    # Python package dependencies
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd microLM
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## End-to-End Model Workflow

This project includes a structured workflow in the `src/` directory to demonstrate training a base model, fine-tuning it, and using both for different tasks.

### Step 1: Train the Base Model

The `src/training.py` script trains a larger language model from scratch on the TinyStories dataset. This creates a "base model" with general text-completion abilities.

```bash
py src/training.py
```
This will save `base_model.pt` and `tokenizer.json` to the `models/` directory.

### Step 2: Fine-Tune the Model

The `src/fine_tune.py` script loads the base model and continues training it on a different dataset (a dialogue summarization dataset) to specialize its behavior for conversation.

```bash
py src/fine_tune.py
```
This saves the new model as `fine_tuned_model.pt` in the `models/` directory.

### Step 3: Run Inference and Chat

You can now use both the base and fine-tuned models.

*   **Base Model Inference**: Use `src/inference.py` to see the raw text generation from the base model. It will continue a prompt with creative, story-like text.

    ```bash
    py src/inference.py
    ```

*   **Chat with the Fine-Tuned Model**: Use `src/chat.py` to have an interactive chat. The responses from this model should be more conversational.

    ```bash
    py src/chat.py
    ```

**Important Note on Chat**: The chat functionality in `src/chat.py` is a **simulation** to demonstrate how a base model's behavior can be altered through fine-tuning. The model is **not** a true instruction-following or conversational AI. It mimics a chat format by continuing a structured prompt.

### Running Unit Tests

To ensure all components are working correctly, you can run the full suite of unit tests:

```bash
python -m unittest discover tests
```


## Configuration

Model and training configurations are managed in `micro_lm/config.py`. You can easily create new configurations or modify existing ones to experiment with different hyperparameters like vocabulary size, block size, learning rate, and more.
Playing with small language models
