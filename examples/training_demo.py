import torch
from torch.utils.data import DataLoader

from micro_lm.model import MicroLM
from micro_lm.config import ModelConfig
from micro_lm.tokenizer import MicroTokenizer
from micro_lm.dataset import PackedDataset


def training_demo():
    """
    Demonstrates a basic training loop for the MicroLM model.

    This function shows how to:
    1. Set up the model, tokenizer, and dataset.
    2. Initialize an optimizer.
    3. Run a simplified training loop for a few steps.
    """
    print("Starting training demo...")

    # 1. Setup
    # Configuration
    config = ModelConfig(
        block_size=128,
        vocab_size=1024,
        n_layer=4,
        n_head=4,
        n_embd=128
    )
    
    # Model and Tokenizer
    model = MicroLM(config)
    tokenizer = MicroTokenizer()

    # Train the tokenizer on a sample of the dataset
    print("Training tokenizer...")
    try:
        from datasets import load_dataset
        dataset_name = "roneneldan/TinyStories"
        training_corpus = load_dataset(dataset_name, split='train', streaming=True).take(1000)
        text_iterator = (item['text'] for item in training_corpus)
        tokenizer.train_from_iterator(text_iterator, vocab_size=config.vocab_size)
        print(f"Tokenizer trained with vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"Could not train tokenizer: {e}")
        # Exit if tokenizer fails, as training cannot proceed
        return

    # Dataset and DataLoader
    dataset = PackedDataset("roneneldan/TinyStories", tokenizer, config.block_size)
    dataloader = DataLoader(dataset, batch_size=8)

    print("\nStep 1: Model, tokenizer, and dataset initialized.")

    # 2. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print("\nStep 2: Optimizer (AdamW) initialized.")

    # 3. Training Loop
    print("\nStep 3: Starting simplified training loop...")
    model.train() # Set model to training mode
    max_steps = 5

    for i, (x, y) in enumerate(dataloader):
        if i >= max_steps:
            break

        # Forward pass
        logits, loss = model(x, y)

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        print(f"Step {i+1}/{max_steps} | Loss: {loss.item():.4f}")

    print("\nTraining loop finished.")

    # Save a dummy checkpoint
    # In a real scenario, you would save the model's state_dict
    # torch.save(model.state_dict(), 'microlm_demo_ckpt.pt')
    print("\nDemonstration of saving a checkpoint would happen here.")

    print("\nTraining demo finished.")


if __name__ == "__main__":
    training_demo()
