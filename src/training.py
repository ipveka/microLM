import os
import torch
import yaml
from torch.utils.data import DataLoader
from micro_lm.model import MicroLM
from micro_lm.config import ModelConfig
from micro_lm.tokenizer import MicroTokenizer
from micro_lm.dataset import PackedDataset


def train_model():
    """
    Trains a larger MicroLM model and saves the model and tokenizer,
    using parameters from config.yaml.
    """
    print("Starting model training process...")

    # Load configuration from YAML file
    with open('config.yaml', 'r') as f:
        config_data = yaml.safe_load(f)

    model_params = config_data['model']
    paths = config_data['paths']
    training_params = config_data['training']

    # 1. Configuration for the model
    config = ModelConfig(
        block_size=model_params['block_size'],
        vocab_size=model_params['vocab_size'],
        n_layer=model_params['n_layer'],
        n_head=model_params['n_head'],
        n_embd=model_params['n_embd']
    )

    # Create directories for models if it doesn't exist
    os.makedirs(os.path.dirname(paths['base_model']), exist_ok=True)
    model_save_path = paths['base_model']
    tokenizer_save_path = paths['tokenizer']

    # 2. Model and Tokenizer Initialization
    model = MicroLM(config)
    tokenizer = MicroTokenizer()

    # 3. Tokenizer Training
    print("Training tokenizer...")
    try:
        from datasets import load_dataset
        dataset_name = training_params['dataset_name']
        training_corpus = load_dataset(
            dataset_name, 
            split='train', 
            streaming=True
        ).take(training_params['tokenizer_training_sample'])
        text_iterator = (item['text'] for item in training_corpus)
        tokenizer.train_from_iterator(text_iterator, vocab_size=config.vocab_size)
        tokenizer.save(tokenizer_save_path)
        print(f"Tokenizer trained and saved to {tokenizer_save_path}")
    except Exception as e:
        print(f"Could not train tokenizer: {e}")
        return

    # 4. Dataset and DataLoader
    print("Initializing dataset and dataloader...")
    dataset = PackedDataset(dataset_name, tokenizer, config.block_size)
    dataloader = DataLoader(dataset, batch_size=training_params['batch_size'])

    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_params['learning_rate'])

    # 6. Training Loop
    print("Starting model training loop...")
    model.train()
    max_steps = training_params['max_steps']

    for i, (x, y) in enumerate(dataloader):
        if i >= max_steps:
            break

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (i % 10 == 0) or (i == max_steps - 1):
            print(f"Step {i+1}/{max_steps} | Loss: {loss.item():.4f}")

    print("\nTraining loop finished.")

    # 7. Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    print("\nTraining process complete.")


if __name__ == "__main__":
    train_model()
