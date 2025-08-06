import torch

from micro_lm.model import MicroLM
from micro_lm.config import ModelConfig
from micro_lm.tokenizer import MicroTokenizer
from micro_lm.utils import setup_device_and_model, print_device_info


def text_generation_demo():
    """
    Demonstrates how to generate text using a pre-trained MicroLM model.

    This function shows how to:
    1. Load a model and tokenizer (or initialize for demo purposes).
    2. Create a starting prompt.
    3. Tokenize the prompt.
    4. Generate new tokens using the model.
    5. Decode the generated tokens back into text.
    """
    print("Starting text generation demo...")

    # For demonstration, we'll initialize a new model and tokenizer.
    # In a real use case, you would load your trained model and tokenizer.
    print_device_info()
    
    config = ModelConfig(vocab_size=512) # Use a small vocab for this example
    model = MicroLM(config)
    model, device = setup_device_and_model(model, device='auto')
    tokenizer = MicroTokenizer()

    # Train the tokenizer on a sample of the dataset
    print("Training tokenizer for the demo...")
    try:
        from datasets import load_dataset
        dataset_name = "roneneldan/TinyStories"
        training_corpus = load_dataset(dataset_name, split='train', streaming=True).take(1000)
        text_iterator = (item['text'] for item in training_corpus)
        tokenizer.train_from_iterator(text_iterator, vocab_size=config.vocab_size)
        print(f"Tokenizer trained with vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"Could not train tokenizer: {e}")
        return

    # Set the model to evaluation mode
    model.eval()

    print("\nStep 1: Model and tokenizer ready.")

    # 2. Create a starting prompt
    prompt = "Once upon a time" 
    print(f"\nStep 2: Starting prompt is: '{prompt}'")

    # 3. Tokenize the prompt
    # The model expects a batch of token indices, so we add a batch dimension.
    encoded_prompt = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
    encoded_prompt = encoded_prompt.to(device)  # Move to device
    print("\nStep 3: Prompt tokenized successfully.")
    print(f"Tokenized input shape: {encoded_prompt.shape}")

    # 4. Generate new tokens
    print("\nStep 4: Generating new tokens...")
    max_new_tokens = 50
    
    # We assume the model has a 'generate' method similar to Hugging Face models.
    # This is a common pattern for autoregressive generation.
    try:
        with torch.no_grad():
            generated_tokens = model.generate(encoded_prompt, max_new_tokens=max_new_tokens)
        
        print(f"Generated {max_new_tokens} new tokens.")

        # 5. Decode the generated tokens
        print("\nStep 5: Decoding the generated text...")
        generated_text = tokenizer.decode(generated_tokens[0].tolist())

        print("\n--- Generated Text ---")
        print(generated_text)
        print("----------------------")

    except AttributeError:
        print("\nError: The model does not have a 'generate' method.")
        print("This demo assumes a 'generate' method exists for inference.")
        print("You may need to implement this method in 'micro_lm/model.py' or 'micro_lm/inference.py'.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during generation: {e}")

    print("\nText generation demo finished.")


if __name__ == "__main__":
    text_generation_demo()
