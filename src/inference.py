import torch
import os
import yaml
from micro_lm.model import MicroLM
from micro_lm.config import ModelConfig
from micro_lm.tokenizer import MicroTokenizer


def run_inference():
    """
    Loads the base model and generates text from a prompt,
    using parameters from config.yaml.
    """
    print("Starting inference with the base model...")

    # Load configuration from YAML file
    with open('config.yaml', 'r') as f:
        config_data = yaml.safe_load(f)

    model_params = config_data['model']
    paths = config_data['paths']
    inference_params = config_data['inference']

    model_path = paths['base_model']
    tokenizer_path = paths['tokenizer']

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print("Error: Base model or tokenizer not found. Please run training.py first.")
        return

    # 1. Configuration must match the saved model
    config = ModelConfig(
        block_size=model_params['block_size'],
        vocab_size=model_params['vocab_size'],
        n_layer=model_params['n_layer'],
        n_head=model_params['n_head'],
        n_embd=model_params['n_embd']
    )

    # 2. Load the model and tokenizer
    model = MicroLM(config)
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set to evaluation mode
    tokenizer = MicroTokenizer(model_path=tokenizer_path)
    print("Base model and tokenizer loaded.")

    # 3. Define a prompt
    prompt = inference_params['prompt']
    print(f'\nPrompt: "{prompt}"')

    # 4. Generate text
    encoded_prompt = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)

    print("\nGenerating text...")
    print("-----------------------------------------------------")
    with torch.no_grad():
        generated_tokens = model.generate(encoded_prompt, max_new_tokens=inference_params['max_new_tokens'])

    response_tokens = generated_tokens[0].tolist()
    response_text = tokenizer.decode(response_tokens)

    # The generated text includes the prompt, so we can print it all together
    print(response_text)
    print("-----------------------------------------------------")

    print("\nInference complete.")


if __name__ == "__main__":
    run_inference()
