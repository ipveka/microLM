import unittest
from unittest.mock import MagicMock, patch, call
import torch

from micro_lm.train import Trainer
from micro_lm.config import TrainConfig


class TestTrainer(unittest.TestCase):

    def setUp(self):
        """Set up mock objects for testing the Trainer."""
        self.mock_config = MagicMock(spec=TrainConfig)
        self.mock_config.device = 'cpu'
        self.mock_config.compile = False
        self.mock_config.weight_decay = 0.1
        self.mock_config.learning_rate = 1e-4
        self.mock_config.beta1 = 0.9
        self.mock_config.beta2 = 0.99
        self.mock_config.grad_clip = 1.0
        self.mock_config.max_iters = 10
        self.mock_config.log_interval = 1
        self.mock_config.eval_interval = 5
        self.mock_config.eval_iters = 2
        self.mock_config.wandb_log = False
        self.mock_config.out_dir = 'test_out'

        self.mock_model = MagicMock()
        # Mock the forward pass to return a mock loss tensor
        self.mock_loss = MagicMock(spec=torch.Tensor)
        self.mock_loss.item.return_value = 0.1234 # Mock the item() call to return a float
        self.mock_model.return_value = (MagicMock(), self.mock_loss)
        self.mock_model.configure_optimizers.return_value = MagicMock(spec=torch.optim.AdamW)
        
        # Mock dataloaders to be iterable and return mock data
        self.mock_train_loader = MagicMock()
        self.mock_train_loader.__iter__.return_value = iter([(torch.randn(2, 4), torch.randn(2, 4)) for _ in range(20)])
        self.mock_val_loader = MagicMock()
        self.mock_val_loader.__iter__.return_value = iter([(torch.randn(2, 4), torch.randn(2, 4)) for _ in range(20)])

    def test_initialization(self):
        """Test that the Trainer initializes correctly."""
        trainer = Trainer(self.mock_config, self.mock_model, self.mock_train_loader, self.mock_val_loader)
        
        self.mock_model.to.assert_called_once_with('cpu')
        self.mock_model.configure_optimizers.assert_called_once()
        self.assertIsNotNone(trainer.optimizer)

    def test_train_step(self):
        """Test a single training step logic."""
        # Limit max_iters to 1 to test a single step
        self.mock_config.max_iters = 1
        trainer = Trainer(self.mock_config, self.mock_model, self.mock_train_loader, self.mock_val_loader)
        self.mock_model.reset_mock() # Reset mock after initialization
        
        trainer.run()

        # Check that the core training operations were called
        trainer.optimizer.zero_grad.assert_called_once()
        self.mock_loss.backward.assert_called_once()
        trainer.optimizer.step.assert_called_once()

    @patch('micro_lm.train.Trainer.save_checkpoint')
    def test_run_loop_eval_and_checkpoint(self, mock_save_checkpoint):
        """Test that evaluation and checkpointing are triggered."""
        self.mock_config.max_iters = 6
        self.mock_config.eval_interval = 5
        self.mock_config.eval_iters = 2

        # Mock evaluate to return a decreasing loss
        trainer = Trainer(self.mock_config, self.mock_model, self.mock_train_loader, self.mock_val_loader)
        trainer.evaluate = MagicMock(side_effect=[0.5, 0.2]) # First eval loss 0.5, second 0.2

        trainer.run()

        # Assert that evaluate was called at the correct intervals (iter 0 and 5)
        self.assertEqual(trainer.evaluate.call_count, 2)
        # Assert that checkpoint was saved once (it's skipped on the first eval at iter 0)
        mock_save_checkpoint.assert_called_once()


if __name__ == '__main__':
    unittest.main()

