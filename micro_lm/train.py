import os
import time
import math
import torch
from tqdm import tqdm
import wandb

from .model import MicroLM
from .config import TrainConfig, get_config
from .dataset import get_dataloader
from .tokenizer import MicroTokenizer

class Trainer:
    """
    A class to handle the training and evaluation of the MicroLM model.
    """
    def __init__(self, config: TrainConfig, model: MicroLM, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader):
        """
        Initializes the Trainer.

        Args:
            config (TrainConfig): The training configuration.
            model (MicroLM): The model to train.
            train_loader (DataLoader): The DataLoader for the training set.
            val_loader (DataLoader): The DataLoader for the validation set.
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2))
        
        self.device = config.device
        self.model.to(self.device)

        # Optional: compile the model for a speedup
        if config.compile:
            print("Compiling the model... (this may take a minute)")
            self.model = torch.compile(self.model)

        self.iter_num = 0
        self.best_val_loss = 1e9

    def run(self):
        """
        Runs the main training loop.
        """
        train_iterator = iter(self.train_loader)
        
        pbar = tqdm(range(self.config.max_iters), desc="Training")
        for self.iter_num in pbar:
            t0 = time.time()

            # Fetch the next batch
            try:
                x, y = next(train_iterator)
            except StopIteration:
                # Epoch finished, start a new one
                train_iterator = iter(self.train_loader)
                x, y = next(train_iterator)

            x, y = x.to(self.device), y.to(self.device)

            # Forward pass and loss calculation
            logits, loss = self.model(x, y)

            # Backward pass and optimization
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            # Log and evaluate
            if self.iter_num % self.config.log_interval == 0:
                dt = time.time() - t0
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "step_time": f"{dt*1000:.2f}ms"
                })
                if self.config.wandb_log:
                    wandb.log({"train/loss": loss.item(), "iter": self.iter_num})


            if self.iter_num % self.config.eval_interval == 0:
                val_loss = self.evaluate()
                print(f"Iter {self.iter_num}: Val loss {val_loss:.4f}")
                if self.config.wandb_log:
                    wandb.log({"val/loss": val_loss, "iter": self.iter_num})

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.iter_num > 0:
                        self.save_checkpoint('best_model.pt')

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        val_iterator = iter(self.val_loader)
        losses = []
        for _ in range(self.config.eval_iters):
            try:
                x, y = next(val_iterator)
            except StopIteration:
                val_iterator = iter(self.val_loader)
                x, y = next(val_iterator)
                
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            losses.append(loss.item())
        
        self.model.train()
        return torch.tensor(losses).mean().item()

    def save_checkpoint(self, filename: str):
        """
        Saves a model checkpoint.
        """
        if not os.path.exists(self.config.out_dir):
            os.makedirs(self.config.out_dir)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
        }
        path = os.path.join(self.config.out_dir, filename)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")


if __name__ == '__main__':
    # --- Example Usage ---
    # This block demonstrates how to set up and run the trainer.
    # Note: This is a placeholder and will be moved to `examples/`.
    
    # 1. Get configuration
    config = get_config('tinystories')
    config.wandb_log = False # Disable wandb for this example
    config.max_iters = 100 # Run for a few iterations for demonstration
    config.eval_iters = 10
    config.log_interval = 10
    config.eval_interval = 50

    # 2. Initialize tokenizer
    tokenizer = MicroTokenizer.from_pretrained('gpt2') # Using GPT-2 for simplicity

    # 3. Create DataLoaders
    train_loader = get_dataloader(
        dataset_name='roneneldan/TinyStories',
        tokenizer=tokenizer,
        config=config.model_config,
        batch_size=config.batch_size,
        split='train'
    )
    val_loader = get_dataloader(
        dataset_name='roneneldan/TinyStories',
        tokenizer=tokenizer,
        config=config.model_config,
        batch_size=config.batch_size,
        split='validation'
    )

    # 4. Initialize model
    model = MicroLM(config.model_config)

    # 5. Initialize and run trainer
    trainer = Trainer(config, model, train_loader, val_loader)
    print("Starting trainer example run...")
    trainer.run()
    print("Trainer example run finished.")
