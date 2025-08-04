import sys
import os
import unittest
import tempfile

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from micro_lm.tokenizer import MicroTokenizer

class TestMicroTokenizer(unittest.TestCase):
    """
    Unit tests for the MicroTokenizer class.
    """
    def setUp(self):
        """
        Set up a temporary directory and a dummy text file for training.
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dummy_text_file = os.path.join(self.temp_dir.name, "dummy_text.txt")
        self.tokenizer_save_path = os.path.join(self.temp_dir.name, "test_tokenizer.json")

        # Create a dummy text file with some content
        with open(self.dummy_text_file, "w", encoding="utf-8") as f:
            f.write("This is a test sentence.\n")
            f.write("This is another sentence for our tokenizer.\n")
            f.write("Testing tokenization is important.\n")

    def tearDown(self):
        """
        Clean up the temporary directory after tests are done.
        """
        self.temp_dir.cleanup()

    def test_from_pretrained(self):
        """
        Test loading a pretrained tokenizer.
        """
        tokenizer = MicroTokenizer.from_pretrained('gpt2')
        self.assertIsNotNone(tokenizer)
        self.assertGreater(tokenizer.vocab_size, 50000) # GPT-2 has >50k tokens
        
        text = "Hello, world!"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(text, decoded)

    def test_train_save_load(self):
        """
        Test training a new tokenizer, saving it, and loading it back.
        """
        # 1. Train a new tokenizer
        vocab_size = 100
        special_tokens = ["[UNK]", "[PAD]", "<BOS>", "<EOS>"]
        trained_tokenizer = MicroTokenizer()
        trained_tokenizer.train(
            files=[self.dummy_text_file],
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )
        self.assertIsNotNone(trained_tokenizer)
        self.assertLessEqual(trained_tokenizer.vocab_size, vocab_size)

        # 2. Test encoding and decoding with the new tokenizer
        text = "This is a test sentence."
        encoded = trained_tokenizer.encode(text)
        self.assertIsInstance(encoded, list)
        self.assertTrue(all(isinstance(i, int) for i in encoded))
        
        decoded = trained_tokenizer.decode(encoded)
        self.assertEqual(text, decoded.strip())

        # 3. Save the tokenizer
        trained_tokenizer.save(self.tokenizer_save_path)
        self.assertTrue(os.path.exists(self.tokenizer_save_path))

        # 4. Load the tokenizer back
        loaded_tokenizer = MicroTokenizer(self.tokenizer_save_path)
        self.assertIsNotNone(loaded_tokenizer)
        self.assertEqual(loaded_tokenizer.vocab_size, trained_tokenizer.vocab_size)

        # 5. Verify the loaded tokenizer works
        encoded_loaded = loaded_tokenizer.encode(text)
        self.assertEqual(encoded, encoded_loaded)
        
        decoded_loaded = loaded_tokenizer.decode(encoded_loaded)
        self.assertEqual(text, decoded_loaded.strip())

if __name__ == '__main__':
    unittest.main()
