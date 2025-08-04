import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from tqdm import tqdm
import logging

from .tokenizer import MicroTokenizer
from .config import ModelConfig

# Set up basic logging
logging.basicConfig(level=logging.INFO)

class PackedDataset(IterableDataset):
    """
    An IterableDataset that streams and packs data from a HuggingFace dataset.

    This dataset is designed for autoregressive language modeling. It fetches
    text from a dataset, tokenizes it, and concatenates the tokens into a
    single stream. It then yields chunks of a specified block size.
    """
    def __init__(self, dataset_name: str, tokenizer: MicroTokenizer, block_size: int, split: str = 'train'):
        """
        Args:
            dataset_name (str): The name of the dataset on HuggingFace Hub.
            tokenizer (MicroTokenizer): The tokenizer to use.
            block_size (int): The size of each chunk of text to yield.
            split (str): The dataset split to use (e.g., 'train', 'validation').
        """
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.split = split
        
        # Load the dataset in streaming mode
        # Load the dataset once to find the text column
        try:
            self.dataset = load_dataset(self.dataset_name, split=self.split, streaming=True)
            self.text_column = self._find_text_column()
        except Exception as e:
            logging.error(f"Failed to load dataset '{self.dataset_name}' with split '{self.split}'. Error: {e}")
            raise

    def __iter__(self):
        """
        Iterates through the dataset, tokenizes text, and yields packed blocks.
        """
        buffer = []
        # Get a fresh iterator for each epoch
        dataset_iterator = load_dataset(self.dataset_name, split=self.split, streaming=True)

        for sample in dataset_iterator:
            if self.text_column in sample and sample[self.text_column]:
                text = sample[self.text_column]
                encoded_text = self.tokenizer.encode(text)
                buffer.extend(encoded_text)

                while len(buffer) >= self.block_size + 1:
                    x = torch.tensor(buffer[:self.block_size], dtype=torch.long)
                    y = torch.tensor(buffer[1:self.block_size + 1], dtype=torch.long)
                    yield x, y
                    # Move the buffer forward by one block
                    buffer = buffer[self.block_size:]

    def _find_text_column(self) -> str:
        """
        Inspects the dataset features to find the most likely text column.
        Common names are 'text', 'content', 'document'.
        """
        # Take one sample to inspect its structure
        sample = next(iter(self.dataset))
        possible_cols = ['text', 'content', 'document', 'prompt', 'response']
        for col in possible_cols:
            if col in sample:
                return col
        # If no common column is found, guess the first string column
        for col_name, col_type in sample.items():
             if isinstance(col_type, str):
                 logging.warning(f"Could not find a standard text column. Using '{col_name}' as text source.")
                 return col_name
        raise ValueError("Could not determine the text column in the dataset.")


def get_dataloader(dataset_name: str, tokenizer: MicroTokenizer, config: ModelConfig, batch_size: int, split: str = 'train'):
    """
    Creates a DataLoader for a given dataset.

    Args:
        dataset_name (str): The name of the dataset.
        tokenizer (MicroTokenizer): The tokenizer to use.
        config (ModelConfig): The model configuration containing block_size.
        batch_size (int): The batch size for the DataLoader.
        split (str): The dataset split to use.

    Returns:
        A PyTorch DataLoader instance.
    """
    dataset = PackedDataset(dataset_name, tokenizer, config.block_size, split)
    return DataLoader(dataset, batch_size=batch_size)


if __name__ == '__main__':
    # --- Example Usage ---
    print("--- Testing PackedDataset and DataLoader ---")
    
    # 1. Initialize a tokenizer
    # Using a pre-trained one for simplicity
    gpt2_tokenizer = MicroTokenizer.from_pretrained('gpt2')
    
    # 2. Define a model configuration
    # This provides the block_size
    model_config = ModelConfig(block_size=128)
    
    # 3. Get a DataLoader
    # Using a small, well-known dataset for demonstration
    # Note: 'roneneldan/TinyStories' is a good small dataset for testing
    try:
        dataloader = get_dataloader(
            dataset_name='roneneldan/TinyStories',
            tokenizer=gpt2_tokenizer,
            config=model_config,
            batch_size=4,
            split='train'
        )
        
        # 4. Fetch one batch from the DataLoader
        print("Fetching one batch from the DataLoader...")
        x, y = next(iter(dataloader))
        
        print(f"Batch fetched successfully!")
        print(f"Input batch shape (x): {x.shape}")
        print(f"Target batch shape (y): {y.shape}")
        
        print("\nSample input sequence (first item in batch):")
        print(x[0])
        
        print("\nDecoded sample input:")
        print(gpt2_tokenizer.decode(x[0].tolist()))

    except Exception as e:
        print(f"An error occurred during the example run: {e}")
        print("This might be due to network issues or the dataset being unavailable.")
