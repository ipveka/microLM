# microLM: A Compact Transformer Language Model Framework

A lightweight and easy-to-understand framework for training and experimenting with Transformer-based language models. Built with PyTorch.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd microLM
pip install -r requirements.txt
```

## Examples

The `examples/` directory contains a set of scripts to demonstrate the core functionalities of `microLM`. 

### `data_preparation_demo.py`

Demonstrates the full data pipeline, including:
- Initializing the tokenizer.
- Loading a dataset from Hugging Face (`roneneldan/TinyStories`).
- Wrapping the data in the `PackedDataset` for efficient, autoregressive training.
- Using a `DataLoader` to create and inspect a batch.

**To run:**
```bash
python examples/data_preparation_demo.py
```

### `model_inspection_demo.py`

Shows how to instantiate and inspect the `MicroLM` model:
- Creating a `ModelConfig` to define the model's architecture.
- Printing the model summary and architecture.
- Calculating the total number of parameters.

**To run:**
```bash
python examples/model_inspection_demo.py
```

### `training_demo.py`

Provides a minimal working example of a training loop:
- Setting up the model, tokenizer, and dataset.
- Initializing an AdamW optimizer.
- Running a few training steps, including forward pass, loss calculation, and backpropagation.

**To run:**
```bash
python examples/training_demo.py
```

### `text_generation_demo.py`

Demonstrates how to generate text with a model:
- Loading a model and tokenizer.
- Encoding a text prompt.
- Using the model's `generate` method to produce new text.
- Decoding the output tokens back into a human-readable string.

**To run:**
```bash
python examples/text_generation_demo.py
```

### `chat_demo.py`

An example of an interactive chat session with the model:
- Loading a model and tokenizer.
- Entering a loop that accepts user input.
- Generating and printing the model's response in real-time.

**To run:**
```bash
python examples/chat_demo.py
```


`microLM` is a lightweight, modular, and well-tested framework for building, training, and running GPT-style language models from scratch. It is designed to be easy to understand, modify, and extend, making it an ideal tool for educational purposes, research, and hobbyist projects. The entire framework is optimized to run efficiently on standard CPUs and low-end GPUs.

## Features

- **Modular Architecture**: Code is cleanly separated into modules for tokenization, data handling, model architecture, training, and inference.
- **Custom & Pretrained Tokenizers**: Supports loading pretrained HuggingFace tokenizers (like GPT-2) or training your own custom BPE tokenizer on your data.
- **Efficient Data Handling**: Uses streaming datasets to handle large text corpora without requiring massive amounts of RAM.
- **Comprehensive Unit Tests**: A full suite of unit tests ensures the reliability and correctness of all core components.
- **Clear Configuration**: A simple, dataclass-based configuration system makes it easy to define and manage model and training parameters.
- **Example Scripts**: Includes ready-to-run examples for training a model and generating text.

## Project Structure

```
. 
├── examples/           # Example scripts (e.g., train_tinystories.py)
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
