import unittest
from unittest.mock import patch, MagicMock
import torch
from torch.utils.data import DataLoader

from micro_lm.dataset import PackedDataset, get_dataloader
from micro_lm.tokenizer import MicroTokenizer
from micro_lm.config import ModelConfig


class TestPackedDataset(unittest.TestCase):

    def setUp(self):
        """Set up a mock tokenizer and mock data for testing."""
        self.mock_tokenizer = MagicMock(spec=MicroTokenizer)
        # Simple encoding: map characters to their ASCII values
        self.mock_tokenizer.encode.side_effect = lambda text: [ord(c) for c in text]

        self.block_size = 10
        self.mock_config = ModelConfig(block_size=self.block_size)

        # Mock data stream from HuggingFace datasets
        self.mock_data = [
            {'text': 'This is the first sentence.'},
            {'text': 'Here is another one.'},
            {'text': 'And a third for good measure.'}
        ]

    @patch('micro_lm.dataset.load_dataset')
    def test_initialization(self, mock_load_dataset):
        """Test that the dataset initializes correctly and calls load_dataset."""
        mock_load_dataset.return_value = iter(self.mock_data)

        dataset = PackedDataset(
            dataset_name='dummy_dataset',
            tokenizer=self.mock_tokenizer,
            block_size=self.block_size,
            split='train'
        )

        mock_load_dataset.assert_called_once_with('dummy_dataset', split='train', streaming=True)
        self.assertIsNotNone(dataset)

    @patch('micro_lm.dataset.load_dataset')
    def test_iteration_and_packing(self, mock_load_dataset):
        """Test that the dataset yields correctly shaped and shifted tensors."""
        # Make the mock return a new iterator each time it's called
        mock_load_dataset.side_effect = lambda *args, **kwargs: iter(self.mock_data)

        dataset = PackedDataset(
            dataset_name='dummy_dataset',
            tokenizer=self.mock_tokenizer,
            block_size=self.block_size,
            split='train'
        )

        # Combine all text to know the expected token stream
        full_text = ''.join(d['text'] for d in self.mock_data)
        full_tokens = [ord(c) for c in full_text]

        # Get the first packed item
        x, y = next(iter(dataset))

        self.assertEqual(x.shape, (self.block_size,))
        self.assertEqual(y.shape, (self.block_size,))

        # Check that y is x shifted by one
        expected_x = torch.tensor(full_tokens[:self.block_size], dtype=torch.long)
        expected_y = torch.tensor(full_tokens[1:self.block_size + 1], dtype=torch.long)

        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

    @patch('micro_lm.dataset.load_dataset')
    def test_text_column_is_found(self, mock_load_dataset):
        """Test that the text_column attribute is correctly identified on initialization."""
        # Test with a standard 'text' column
        mock_load_dataset.return_value = iter([{'text': 'hello'}])
        dataset = PackedDataset('dummy', self.mock_tokenizer, 10)
        self.assertEqual(dataset.text_column, 'text')

        # Test with a non-standard 'content' column
        mock_load_dataset.return_value = iter([{'content': 'world'}])
        dataset = PackedDataset('dummy', self.mock_tokenizer, 10)
        self.assertEqual(dataset.text_column, 'content')

    @patch('micro_lm.dataset.PackedDataset')
    def test_get_dataloader(self, MockPackedDataset):
        """Test that the get_dataloader function returns a configured DataLoader."""
        # Arrange
        mock_instance = MagicMock()
        MockPackedDataset.return_value = mock_instance
        batch_size = 4

        # Act
        dataloader = get_dataloader(
            dataset_name='dummy_dataset',
            tokenizer=self.mock_tokenizer,
            config=self.mock_config,
            batch_size=batch_size,
            split='validation'
        )

        # Assert
        MockPackedDataset.assert_called_once_with(
            'dummy_dataset', self.mock_tokenizer, self.block_size, 'validation'
        )
        self.assertIsInstance(dataloader, DataLoader)
        self.assertEqual(dataloader.batch_size, batch_size)


if __name__ == '__main__':
    unittest.main()
