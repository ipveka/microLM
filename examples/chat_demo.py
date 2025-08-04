import torch

from micro_lm.model import MicroLM
from micro_lm.config import ModelConfig
from micro_lm.tokenizer import MicroTokenizer


def chat_demo():
    """
    Demonstrates an interactive chat session with a MicroLM model.

    This function shows how to:
    1. Load a pre-trained model (or initialize a new one for demo).
    2. Run an interactive loop to get user input.
    3. Format the input as a prompt for the model.
    4. Generate and stream the model's response.
    """
    print("Starting interactive chat demo...")

    # 1. Load model and tokenizer
    # For a real chat, you would load a model fine-tuned for conversations.
    config = ModelConfig(vocab_size=1024)
    model = MicroLM(config)
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

    model.eval() # Set to evaluation mode

    print("Model and tokenizer ready. Type 'q' to end the chat.")
    print("-----------------------------------------------------")

    # 2. Interactive Loop
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'q':
            print("Exiting chat. Goodbye!")
            break

        # 3. Format the prompt
        # A common chat format uses special tokens for roles.
        # We'll simulate this, though the base model isn't specifically trained for it.
        prompt = f"<user>\n{user_input}\n<assistant>\n"
        
        # 4. Generate response
        encoded_prompt = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
        
        print("Assistant: ", end="", flush=True)
        
        try:
            # We assume a 'generate' method exists.
            # For a true streaming effect, the generate method would need to yield tokens.
            # Here, we simulate it by generating the full response and then printing.
            with torch.no_grad():
                generated_tokens = model.generate(encoded_prompt, max_new_tokens=60)
            
            response_tokens = generated_tokens[0].tolist()
            response_text = tokenizer.decode(response_tokens)
            
            # The model output includes the prompt, so we remove it.
            clean_response = response_text.replace(prompt, "")
            print(clean_response)

        except AttributeError:
            print("\nError: Model lacks a 'generate' method for inference.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break
            
    print("-----------------------------------------------------")
    print("Chat demo finished.")


if __name__ == "__main__":
    chat_demo()
