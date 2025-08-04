import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer

class MicroTokenizer:
    """
    A simple wrapper around the Hugging Face tokenizers library.
    Handles training, encoding, and decoding.
    """
    def __init__(self, model_path: str = None):
        """
        Initializes the tokenizer.

        Args:
            model_path (str, optional): Path to a saved tokenizer model file. 
                                      If None, the tokenizer is uninitialized.
        """
        if model_path and os.path.exists(model_path):
            self.tokenizer = Tokenizer.from_file(model_path)
        else:
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = Whitespace()

    @classmethod
    def from_pretrained(cls, model_name: str):
        """
        Loads a pretrained tokenizer from the Hugging Face Hub.

        Args:
            model_name (str): The name of the pretrained model (e.g., 'gpt2').

        Returns:
            MicroTokenizer: A new instance of the tokenizer.
        """
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_obj = cls()
        tokenizer_obj.tokenizer = hf_tokenizer.backend_tokenizer
        return tokenizer_obj

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return self.tokenizer.get_vocab_size()

    def train_from_iterator(self, iterator, vocab_size: int, special_tokens=None):
        """
        Trains the tokenizer on a text iterator.

        Args:
            iterator: An iterator that yields strings (e.g., lines of a text file).
            vocab_size (int): The desired size of the vocabulary.
            special_tokens (list, optional): A list of special tokens to add.
                                           Defaults to ['[UNK]', '[PAD]', '[SOS]', '[EOS]'].
        """
        if special_tokens is None:
            special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
        
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        self.tokenizer.train_from_iterator(iterator, trainer=trainer)

    def train(self, files: list[str], vocab_size: int, special_tokens=None):
        """
        Trains the tokenizer from a list of text files.

        Args:
            files (list[str]): A list of paths to text files.
            vocab_size (int): The desired size of the vocabulary.
            special_tokens (list, optional): A list of special tokens to add.
        """
        def file_iterator():
            for path in files:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        yield line

        self.train_from_iterator(file_iterator(), vocab_size, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encodes a string into a list of token IDs."""
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """Decodes a list of token IDs back into a string."""
        decoded_text = self.tokenizer.decode(ids)
        # Basic post-processing to handle extra spaces around punctuation
        decoded_text = decoded_text.replace(' .', '.').replace(' ,', ',').replace(' ?', '?').replace(' !', '!')
        return decoded_text

    def save(self, path: str):
        """
        Saves the tokenizer model to a file.

        Args:
            path (str): The path to save the tokenizer file (e.g., 'tokenizer.json').
        """
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.tokenizer.save(path)
