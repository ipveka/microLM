import os
import torch
import yaml
from torch.utils.data import DataLoader
from micro_lm.model import MicroLM
from micro_lm.config import ModelConfig
from micro_lm.tokenizer import MicroTokenizer
from micro_lm.dataset import PackedDataset


def fine_tune_model():
    """
    Loads the base model and fine-tunes it on a conversational dataset,
    using parameters from config.yaml.
    """
    print("Starting model fine-tuning process...")

    # Load configuration from YAML file
    with open('config.yaml', 'r') as f:
        config_data = yaml.safe_load(f)

    model_params = config_data['model']
    paths = config_data['paths']
    fine_tuning_params = config_data['fine_tuning']

    base_model_path = paths['base_model']
    tokenizer_path = paths['tokenizer']
    fine_tuned_model_save_path = paths['fine_tuned_model']

    if not os.path.exists(base_model_path) or not os.path.exists(tokenizer_path):
        print(f"Error: Base model or tokenizer not found. Please run training.py first.")
        return

    # 1. Configuration must match the base model
    config = ModelConfig(
        block_size=model_params['block_size'],
        vocab_size=model_params['vocab_size'],
        n_layer=model_params['n_layer'],
        n_head=model_params['n_head'],
        n_embd=model_params['n_embd']
    )

    # 2. Load the base model and tokenizer
    model = MicroLM(config)
    model.load_state_dict(torch.load(base_model_path))
    tokenizer = MicroTokenizer(model_path=tokenizer_path)
    print("Base model and tokenizer loaded.")

    # 3. Prepare the fine-tuning dataset
    print("Initializing fine-tuning dataset...")
    fine_tune_dataset_name = fine_tuning_params['dataset_name']
    dataset = PackedDataset(fine_tune_dataset_name, tokenizer, config.block_size, split='train')
    dataloader = DataLoader(dataset, batch_size=fine_tuning_params['batch_size'])

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=fine_tuning_params['learning_rate'])

    # 5. Fine-tuning Loop
    print("Starting fine-tuning loop...")
    model.train()
    max_steps = fine_tuning_params['max_steps']

    for i, (x, y) in enumerate(dataloader):
        if i >= max_steps:
            break

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (i % 10 == 0) or (i == max_steps - 1):
            print(f"Step {i+1}/{max_steps} | Loss: {loss.item():.4f}")

    print("\nFine-tuning loop finished.")

    # 6. Save the fine-tuned model
    torch.save(model.state_dict(), fine_tuned_model_save_path)
    print(f"Fine-tuned model saved to {fine_tuned_model_save_path}")

    print("\nFine-tuning process complete.")


if __name__ == "__main__":
    fine_tune_model()
