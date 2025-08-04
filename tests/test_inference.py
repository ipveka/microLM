import unittest
from unittest.mock import MagicMock, patch
import torch

from micro_lm.inference import Generator
from micro_lm.model import MicroLM
from micro_lm.tokenizer import MicroTokenizer


class TestGenerator(unittest.TestCase):

    def setUp(self):
        """Set up mock objects for testing the Generator."""
        # Mock Model
        self.mock_model = MagicMock(spec=MicroLM)
        mock_config = MagicMock()
        mock_config.block_size = 128
        self.mock_model.config = mock_config
        # The model's forward pass returns mock logits and no loss
        mock_logits = torch.randn(1, 1, 100)  # (B, T, C) where T=1 for inference
        self.mock_model.return_value = (mock_logits, None)
        # Mock device by mocking parameters
        self.mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.randn(2, 2))])

        # Mock Tokenizer
        self.mock_tokenizer = MagicMock(spec=MicroTokenizer)
        self.mock_tokenizer.encode.return_value = [1, 2, 3]  # "Once upon a"
        self.mock_tokenizer.decode.return_value = "Once upon a time"

    def test_initialization(self):
        """Test that the Generator initializes correctly and puts the model in eval mode."""
        Generator(self.mock_model, self.mock_tokenizer)
        self.mock_model.eval.assert_called_once()

    @patch('torch.multinomial')
    def test_generate_produces_correct_length(self, mock_multinomial):
        """Test that generate produces a sequence of the correct length."""
        # Mock multinomial to always return the same next token
        mock_multinomial.return_value = torch.tensor([[5]], dtype=torch.long)

        generator = Generator(self.mock_model, self.mock_tokenizer)

        prompt = "test"
        initial_tokens = [10, 20]
        self.mock_tokenizer.encode.return_value = initial_tokens
        max_new_tokens = 5

        generator.generate(prompt, max_new_tokens=max_new_tokens)

        # The final list of tokens passed to decode should have the correct length
        final_token_list = self.mock_tokenizer.decode.call_args[0][0]
        self.assertEqual(len(final_token_list), len(initial_tokens) + max_new_tokens)

    @patch('torch.topk')
    def test_top_k_is_applied(self, mock_topk):
        """Test that top_k sampling logic is applied when the parameter is provided."""
        # To check the logic inside, we need a value to return from topk
        mock_topk.return_value = (torch.randn(1, 10), torch.randn(1, 10))
        generator = Generator(self.mock_model, self.mock_tokenizer)

        # Test with top_k
        generator.generate("test", max_new_tokens=2, top_k=10)
        mock_topk.assert_called()

        # Test without top_k
        mock_topk.reset_mock()
        generator.generate("test", max_new_tokens=2, top_k=None)
        mock_topk.assert_not_called()


if __name__ == '__main__':
    unittest.main()
