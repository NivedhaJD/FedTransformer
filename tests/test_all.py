import sys
import os
import unittest
import torch

# Add the project root to sys path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import CONFIG
from network.dataset import ISACDataset, get_dataloader, NUM_FEATURES
from network.client import ISACClient, build_clients
from federated_learning.models.transformer import ISACTransformer
from federated_learning.aggregation.fedavg import federated_average

class TestFedTransformer(unittest.TestCase):
    def test_config(self):
        self.assertIn("num_clients", CONFIG)
        self.assertIn("learning_rate", CONFIG)
        
    def test_dataset(self):
        # Extremely small test dataset to verify mechanics
        dl = get_dataloader(node_id=0, num_samples=10, seq_len=10, batch_size=2)
        X, y = next(iter(dl))
        self.assertEqual(X.shape, (2, 10, NUM_FEATURES))
        self.assertEqual(y.shape, (2,))
        
    def test_transformer_model(self):
        model = ISACTransformer(input_dim=NUM_FEATURES, d_model=16, num_heads=2, num_layers=1, num_classes=4)
        X = torch.randn(2, 10, NUM_FEATURES)
        out = model(X)
        self.assertEqual(out.shape, (2, 4))
        
    def test_fedavg(self):
        # Create dummy weights simulating 2 clients with equal dataset size
        w1 = {"embed": torch.ones(2, 2)}
        w2 = {"embed": torch.ones(2, 2) * 3}
        results = [
            {"state_dict": w1, "num_samples": 10},
            {"state_dict": w2, "num_samples": 10}
        ]
        agg = federated_average(results)
        # 10/20 * 1 + 10/20 * 3 = 2
        self.assertTrue(torch.allclose(agg["embed"], torch.ones(2, 2) * 2))

if __name__ == "__main__":
    unittest.main()
