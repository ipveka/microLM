import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model import MicroLM
from .tokenizer import MicroTokenizer
from .config import ModelConfig

class Generator:
    """
    A class for generating text using a trained MicroLM model.
    """
    def __init__(self, model: MicroLM, tokenizer: MicroTokenizer):
        """
        Initializes the Generator.

        Args:
            model (MicroLM): The trained model.
            tokenizer (MicroTokenizer): The tokenizer to use.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int, temperature: float = 1.0, top_k: int = None, top_p: float = None):
        """
        Generates text given a prompt.

        Args:
            prompt (str): The initial text to start generation from.
            max_new_tokens (int): The maximum number of new tokens to generate.
            temperature (float): Controls randomness. Higher values mean more randomness.
            top_k (int): If set, samples from the k most likely next tokens.
            top_p (float): If set, samples from the smallest set of tokens whose cumulative probability exceeds top_p.

        Returns:
            The generated text as a string.
        """
        # Encode the prompt
        idx = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)

        # Generate tokens one by one
        for _ in tqdm(range(max_new_tokens), desc="Generating text"):
            # Crop the context if it exceeds block size
            idx_cond = idx if idx.size(1) <= self.model.config.block_size else idx[:, -self.model.config.block_size:]
            
            # Get the logits from the model
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional: Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Optional: Nucleus (Top-p) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('Inf')

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the new token and continue
            idx = torch.cat((idx, idx_next), dim=1)

        # Decode the generated tokens and return the text
        return self.tokenizer.decode(idx[0].tolist())

def load_model_from_checkpoint(checkpoint_path: str, device='cpu') -> tuple[MicroLM, MicroTokenizer, ModelConfig]:
    """
    Loads a model, tokenizer, and config from a checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config'].model_config
    
    # For simplicity, we assume the tokenizer is saved alongside or is a standard one
    # In a real scenario, you might save the tokenizer path in the checkpoint
    tokenizer = MicroTokenizer.from_pretrained('gpt2') # Or load a custom one
    
    model = MicroLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, tokenizer, config


if __name__ == '__main__':
    # --- Example Usage ---
    # Note: This requires a trained model checkpoint to be present.
    
    checkpoint_path = "out/best_model.pt"
    
    import os
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found. Please train a model first.")
        print("You can run `python examples/train_tinystories.py` to create one.")
    else:
        print(f"--- Loading Model from {checkpoint_path} ---")
        model, tokenizer, config = load_model_from_checkpoint(checkpoint_path)
        
        generator = Generator(model, tokenizer)
        
        prompt = "Once upon a time"
        print(f"\n--- Generating with prompt: '{prompt}' ---")
        
        # Example 1: Default generation (temp=1.0, no top-k/p)
        print("\n--- Default Generation ---")
        generated_text = generator.generate(prompt, max_new_tokens=50)
        print(generated_text)
        
        # Example 2: Creative generation (higher temperature)
        print("\n--- Creative Generation (temp=1.5) ---")
        generated_text_creative = generator.generate(prompt, max_new_tokens=50, temperature=1.5)
        print(generated_text_creative)
        
        # Example 3: Controlled generation (top-k)
        print("\n--- Controlled Generation (top_k=10) ---")
        generated_text_topk = generator.generate(prompt, max_new_tokens=50, top_k=10)
        print(generated_text_topk)
