import torch

from micro_lm.model import MicroLM
from micro_lm.config import ModelConfig


def model_inspection_demo():
    """
    Demonstrates how to inspect the MicroLM model.

    This function shows how to:
    1. Define a model configuration.
    2. Instantiate the model.
    3. Print the model architecture.
    4. Calculate and display the number of parameters.
    5. Perform a dummy forward pass to check input/output shapes.
    """
    print("Starting model inspection demo...")

    # 1. Define a model configuration
    # These are example parameters. Adjust them to scale the model.
    config = ModelConfig(
        block_size=256,
        vocab_size=1024, # Small vocab for demo purposes
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        bias=False
    )
    print("\nStep 1: ModelConfig created:")
    print(config)

    # 2. Instantiate the model
    model = MicroLM(config)
    print("\nStep 2: MicroLM model instantiated successfully.")

    # 3. Print the model architecture
    print("\nStep 3: Model Architecture:")
    print(model)

    # 4. Calculate and display the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nStep 4: Model Parameters:")
    print(f"Total number of parameters: {num_params:,}")

    # 5. Perform a dummy forward pass
    print("\nStep 5: Performing a dummy forward pass...")
    batch_size = 2
    seq_length = config.block_size
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_length))

    # Get model output
    logits, loss = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape (logits): {logits.shape}")
    # Loss is typically not calculated during inference, but the model forward pass returns it.
    # Here we pass targets so loss is not None
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    logits, loss = model(dummy_input, targets)
    print(f"Loss value: {loss.item()}")

    print("\nModel inspection demo finished.")


if __name__ == "__main__":
    model_inspection_demo()
