import torch
import os
import yaml
from micro_lm.model import MicroLM
from micro_lm.config import ModelConfig
from micro_lm.tokenizer import MicroTokenizer


def run_chat():
    """
    Loads the fine-tuned model and starts an interactive chat session,
    using parameters from config.yaml.
    """
    print("Starting chat with the fine-tuned model...")

    # Load configuration from YAML file
    with open('config.yaml', 'r') as f:
        config_data = yaml.safe_load(f)

    model_params = config_data['model']
    paths = config_data['paths']
    chat_params = config_data['chat']

    model_path = paths['fine_tuned_model']
    tokenizer_path = paths['tokenizer']

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print("Error: Fine-tuned model or tokenizer not found. Please run fine_tune.py first.")
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
    print("Fine-tuned model and tokenizer loaded.")

    print(f"\n{chat_params['welcome_message']}")
    print("-----------------------------------------------------")

    # 3. Interactive Loop
    while True:
        user_input = input("You: ")
        if user_input.lower() == chat_params['exit_command']:
            print("Exiting chat. Goodbye!")
            break

        # 4. Format the prompt
        prompt = f"<user>\n{user_input}\n<assistant>\n"

        # 5. Generate response
        encoded_prompt = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)

        print("Assistant: ", end="", flush=True)

        with torch.no_grad():
            generated_tokens = model.generate(encoded_prompt, max_new_tokens=chat_params['max_new_tokens'])

        response_tokens = generated_tokens[0].tolist()
        response_text = tokenizer.decode(response_tokens)

        # The model's output includes the prompt, so we remove it for a clean response
        clean_response = response_text.replace(prompt, "").strip()
        print(clean_response)

    print("-----------------------------------------------------")
    print("Chat session finished.")


if __name__ == "__main__":
    run_chat()
