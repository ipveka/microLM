import unittest
import torch

from micro_lm.model import MicroLM as GPT, CausalSelfAttention, Block, ModelConfig


class TestGPTModel(unittest.TestCase):

    def setUp(self):
        """Set up a small model configuration for all tests."""
        self.config = ModelConfig(
            block_size=64,
            vocab_size=100,
            n_layer=2,
            n_head=2,
            n_embd=32,
            bias=False
        )
        self.batch_size = 4

    def test_causal_self_attention_shape(self):
        """Test the output shape of the CausalSelfAttention module."""
        attention = CausalSelfAttention(self.config)
        # Input shape: (batch_size, sequence_length, embedding_dim)
        x = torch.randn(self.batch_size, self.config.block_size, self.config.n_embd)
        y = attention(x)
        self.assertEqual(y.shape, x.shape)

    def test_block_shape(self):
        """Test the output shape of the Block module."""
        block = Block(self.config)
        x = torch.randn(self.batch_size, self.config.block_size, self.config.n_embd)
        y = block(x)
        self.assertEqual(y.shape, x.shape)

    def test_gpt_model_forward_pass(self):
        """Test a full forward pass of the GPT model, checking output shapes."""
        model = GPT(self.config)
        # Input shape: (batch_size, sequence_length)
        idx = torch.randint(0, self.config.vocab_size, (self.batch_size, self.config.block_size))
        
        # Test with targets
        logits, loss = model(idx, targets=idx)
        self.assertEqual(logits.shape, (self.batch_size, self.config.block_size, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss.item(), float)

        # Test without targets
        logits_no_loss, loss_no_loss = model(idx)
        self.assertEqual(logits_no_loss.shape, (self.batch_size, 1, self.config.vocab_size))
        self.assertIsNone(loss_no_loss)

    def test_causality(self):
        """Verify that the attention mechanism is causal."""
        model = GPT(self.config)
        seq_len = 5
        idx = torch.randint(0, self.config.vocab_size, (1, seq_len))

        # Forward pass with original input
        logits1, _ = model(idx)

        # Change a future token and re-run
        idx_clone = idx.clone()
        idx_clone[0, -1] = (idx_clone[0, -1] + 1) % self.config.vocab_size
        logits2, _ = model(idx_clone)

        # The logits for the first few tokens should be identical
        # because they should not be affected by a change in the last token.
        # We check up to the second-to-last token.
        self.assertTrue(torch.allclose(logits1[:, :-1, :], logits2[:, :-1, :], atol=1e-6))

        # The logits for the last token should be different
        self.assertFalse(torch.allclose(logits1[:, -1, :], logits2[:, -1, :]))


if __name__ == '__main__':
    unittest.main()
