import torch
from torch.utils.data import DataLoader

from micro_lm.tokenizer import MicroTokenizer
from micro_lm.dataset import PackedDataset
from micro_lm.config import ModelConfig


def data_preparation_demo():
    """
    Demonstrates the data preparation pipeline for micro_lm.

    This function shows how to:
    1. Initialize the tokenizer.
    2. Load a dataset from Hugging Face.
    3. Create a packed, iterable dataset for efficient training.
    4. Use a DataLoader to inspect a batch of data.
    """
    print("Starting data preparation demo...")

    # Configuration
    config = ModelConfig(
        block_size=128,  # Small block size for demonstration
        vocab_size=2048, # Example vocab size
    )

    # 1. Initialize and Train the Tokenizer
    print("\nStep 1: Initializing and training the tokenizer...")
    tokenizer = MicroTokenizer()
    dataset_name = "roneneldan/TinyStories"

    # For the demo, we'll train a new tokenizer on a sample of the dataset.
    # In a real scenario, you might save/load a pre-trained tokenizer.
    print(f"Loading a sample from '{dataset_name}' to train the tokenizer...")
    try:
        from datasets import load_dataset
        # Use a small sample to train the tokenizer quickly for the demo
        training_corpus = load_dataset(dataset_name, split='train', streaming=True).take(1000)
        text_iterator = (item['text'] for item in training_corpus)

        # We assume the tokenizer has a training method like this.
        # This method needs to be implemented in `micro_lm/tokenizer.py`.
        tokenizer.train_from_iterator(text_iterator, vocab_size=config.vocab_size)

        print(f"Tokenizer trained. Vocabulary size: {tokenizer.vocab_size}")
        if tokenizer.vocab_size == 0:
            print("\nWarning: Tokenizer vocabulary is still empty after training.")
            print("Please ensure 'train_from_iterator' is implemented correctly in 'micro_lm/tokenizer.py'.")

    except (ImportError, AttributeError, Exception) as e:
        print(f"\nCould not train tokenizer automatically: {e}")
        print("This demo assumes 'micro_lm.tokenizer.MicroTokenizer' has a 'train_from_iterator' method.")
        print("Proceeding with an empty tokenizer, which will likely fail.")

    # 2. Create a PackedDataset
    print(f"\nStep 2: Creating PackedDataset with '{dataset_name}'...")

    # 3. Create a PackedDataset
    # This will stream, tokenize, and pack the data on the fly.
    packed_dataset = PackedDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        block_size=config.block_size,
        split='train'  # Use the training split
    )
    print("PackedDataset created successfully.")

    # 4. Use a DataLoader to inspect a batch
    # This demonstrates how the data would be fed to the model during training.
    print("\nStep 3: Creating DataLoader and inspecting a batch...")
    dataloader = DataLoader(packed_dataset, batch_size=4)

    # Get a single batch from the dataloader
    try:
        x, y = next(iter(dataloader))

        print("\n--- Demo Batch --- ")
        print(f"Batch shape (input): {x.shape}")
        print(f"Batch shape (target): {y.shape}")
        print(f"Data type: {x.dtype}")
        print("\nInput tensor (first 2 sequences):")
        print(x[:2])
        print("\nTarget tensor (first 2 sequences):")
        print(y[:2])
        print("------------------")

        # Decode a sequence to see the text
        print("\nDecoding the first sequence in the batch:")
        decoded_text = tokenizer.decode(x[0].tolist())
        print(f"'\n{decoded_text}\n'")

    except Exception as e:
        print(f"\nAn error occurred while fetching a batch: {e}")
        print("This might be due to network issues or dataset availability.")

    print("\nData preparation demo finished.")


if __name__ == "__main__":
    data_preparation_demo()
