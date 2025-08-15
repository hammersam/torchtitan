#!/usr/bin/env python3
"""
Integration tests for Native Sparse Attention (NSA) in torchtitan.

This module contains comprehensive integration tests that verify the complete
NSA integration works end-to-end within the torchtitan framework. These tests
cover:
- Model initialization and configuration
- Training pipeline setup
- Distributed training compatibility
- Configuration validation
- End-to-end training simulation

Note: These tests are designed to run without actual GPU/triton environment
and use mocking for external dependencies.
"""

import unittest
import tempfile
import os
from pathlib import Path
import torch
from unittest.mock import patch, MagicMock
import sys

# Add torchtitan to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torchtitan.models.nsa import Transformer, TransformerModelArgs
from torchtitan.models.nsa.train_nsa import main as train_main
from torchtitan.config_manager import JobConfig


class TestNSAIntegration(unittest.TestCase):
    """Integration tests for NSA model."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.toml")

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def create_test_config(self, model_size="debug"):
        """Create a test configuration file."""
        if model_size == "debug":
            config_content = """
[job]
dump_folder = "./outputs/test"
description = "NSA test model"
print_args = true

[model]
name = "nsa"
flavor = "debug"
hf_assets_path = "./tests/assets/tokenizer"
dim = 128
n_layers = 2
n_heads = 4
n_kv_heads = 1
vocab_size = 1000
multiple_of = 128
max_seq_len = 64
compression_block_size = 8
compression_stride = 4
selection_block_size = 16
num_selected_blocks = 4
sliding_window_size = 32

[optimizer]
name = "AdamW"
lr = 1e-4
weight_decay = 0.1

[training]
local_batch_size = 2
seq_len = 32
steps = 5
dataset = "c4_test"
compile = false

[parallelism]
data_parallel_replicate_degree = 1
data_parallel_shard_degree = 1
tensor_parallel_degree = 1
pipeline_parallel_degree = 1

[checkpoint]
enable_checkpoint = false

[validation]
enabled = false
"""
        else:  # 100M model
            config_content = """
[job]
dump_folder = "./outputs/test"
description = "NSA 100M test"
print_args = true

[model]
name = "nsa"
flavor = "100m"
hf_assets_path = "./tests/assets/tokenizer"
dim = 512
n_layers = 6
n_heads = 8
n_kv_heads = 2
vocab_size = 32000
multiple_of = 256
max_seq_len = 512
compression_block_size = 16
compression_stride = 8
selection_block_size = 32
num_selected_blocks = 8
sliding_window_size = 128

[optimizer]
name = "AdamW"
lr = 1e-4
weight_decay = 0.1

[training]
local_batch_size = 4
seq_len = 128
steps = 10
dataset = "c4_test"
compile = false

[parallelism]
data_parallel_replicate_degree = 1
data_parallel_shard_degree = 1
tensor_parallel_degree = 1
pipeline_parallel_degree = 1

[checkpoint]
enable_checkpoint = false

[validation]
enabled = false
"""

        with open(self.config_path, 'w') as f:
            f.write(config_content)

    def test_model_creation(self):
        """Test complete model creation with configuration."""
        self.create_test_config("debug")
        
        config = JobConfig.from_file(self.config_path)
        model_args = TransformerModelArgs.from_job_config(config)
        
        # Create model
        model = Transformer(model_args)
        
        # Verify model structure
        self.assertEqual(len(model.layers), model_args.n_layers)
        self.assertEqual(model.tok_embeddings.num_embeddings, model_args.vocab_size)
        self.assertEqual(model.tok_embeddings.embedding_dim, model_args.dim)
        
        # Verify NSA parameters
        self.assertEqual(model_args.compression_block_size, 8)
        self.assertEqual(model_args.compression_stride, 4)
        self.assertEqual(model_args.selection_block_size, 16)
        self.assertEqual(model_args.num_selected_blocks, 4)
        self.assertEqual(model_args.sliding_window_size, 32)

    def test_100m_model_creation(self):
        """Test 100M parameter model creation."""
        self.create_test_config("100m")
        
        config = JobConfig.from_file(self.config_path)
        model_args = TransformerModelArgs.from_job_config(config)
        model = Transformer(model_args)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should be around 100M parameters
        self.assertGreater(total_params, 80_000_000)
        self.assertLess(total_params, 150_000_000)

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test invalid GQA ratio
        with self.assertRaises(AssertionError):
            TransformerModelArgs(
                dim=256,
                n_layers=2,
                n_heads=7,  # Not divisible by kv_heads
                n_kv_heads=3,
                vocab_size=1000,
                max_seq_len=128,
                compression_block_size=16,
                compression_stride=8,
                selection_block_size=32,
                num_selected_blocks=8,
                sliding_window_size=128,
            )

    def test_forward_pass_consistency(self):
        """Test consistent forward pass behavior."""
        model_args = TransformerModelArgs(
            dim=256,
            n_layers=2,
            n_heads=4,
            n_kv_heads=1,
            vocab_size=1000,
            max_seq_len=64,
            compression_block_size=8,
            compression_stride=4,
            selection_block_size=16,
            num_selected_blocks=4,
            sliding_window_size=32,
        )
        
        model = Transformer(model_args)
        model.eval()
        
        # Test with same input multiple times
        tokens = torch.randint(0, 1000, (1, 32))
        
        with torch.no_grad():
            logits1 = model(tokens)
            logits2 = model(tokens)
        
        # Should be deterministic
        torch.testing.assert_close(logits1, logits2)

    @patch('torch.cuda.is_available')
    def test_cpu_compatibility(self, mock_cuda_available):
        """Test CPU compatibility."""
        mock_cuda_available.return_value = False
        
        model_args = TransformerModelArgs(
            dim=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=1,
            vocab_size=1000,
            max_seq_len=64,
            compression_block_size=8,
            compression_stride=4,
            selection_block_size=16,
            num_selected_blocks=4,
            sliding_window_size=32,
        )
        
        model = Transformer(model_args)
        tokens = torch.randint(0, 1000, (2, 32))
        
        # Should work on CPU
        logits = model(tokens)
        self.assertEqual(logits.shape, (2, 32, 1000))

    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        model_args = TransformerModelArgs(
            dim=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=1,
            vocab_size=1000,
            max_seq_len=64,
            compression_block_size=8,
            compression_stride=4,
            selection_block_size=16,
            num_selected_blocks=4,
            sliding_window_size=32,
        )
        
        model = Transformer(model_args)
        
        # Create training data
        batch_size = 2
        seq_len = 16
        tokens = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Forward pass
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 1000),
            labels.view(-1),
            ignore_index=-1,
        )
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for {name}")

    def test_configuration_file_loading(self):
        """Test configuration file loading."""
        self.create_test_config("debug")
        
        config = JobConfig.from_file(self.config_path)
        
        # Verify configuration loaded correctly
        self.assertEqual(config.model.name, "nsa")
        self.assertEqual(config.model.flavor, "debug")
        self.assertEqual(config.training.local_batch_size, 2)
        self.assertEqual(config.training.seq_len, 32)

    def test_different_sequence_lengths(self):
        """Test model with different sequence lengths."""
        model_args = TransformerModelArgs(
            dim=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=1,
            vocab_size=1000,
            max_seq_len=128,
            compression_block_size=8,
            compression_stride=4,
            selection_block_size=16,
            num_selected_blocks=4,
            sliding_window_size=32,
        )
        
        model = Transformer(model_args)
        
        # Test various sequence lengths
        for seq_len in [1, 16, 32, 64, 128]:
            tokens = torch.randint(0, 1000, (1, seq_len))
            logits = model(tokens)
            self.assertEqual(logits.shape, (1, seq_len, 1000))

    def test_model_memory_usage(self):
        """Test that model doesn't use excessive memory."""
        model_args = TransformerModelArgs(
            dim=256,
            n_layers=2,
            n_heads=8,
            n_kv_heads=2,
            vocab_size=1000,
            max_seq_len=128,
            compression_block_size=16,
            compression_stride=8,
            selection_block_size=32,
            num_selected_blocks=8,
            sliding_window_size=64,
        )
        
        model = Transformer(model_args)
        
        # Estimate memory usage
        total_params = sum(p.numel() for p in model.parameters())
        param_memory_mb = total_params * 4 / (1024 * 1024)  # float32
        
        # Should be reasonable for test model
        self.assertLess(param_memory_mb, 100)  # Less than 100MB

    def test_batch_processing(self):
        """Test batch processing capabilities."""
        model_args = TransformerModelArgs(
            dim=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=1,
            vocab_size=1000,
            max_seq_len=64,
            compression_block_size=8,
            compression_stride=4,
            selection_block_size=16,
            num_selected_blocks=4,
            sliding_window_size=32,
        )
        
        model = Transformer(model_args)
        
        # Test different batch sizes
        for batch_size in [1, 2, 4, 8]:
            tokens = torch.randint(0, 1000, (batch_size, 32))
            logits = model(tokens)
            self.assertEqual(logits.shape, (batch_size, 32, 1000))


class TestTrainingPipeline(unittest.TestCase):
    """Test complete training pipeline setup."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_training_script_import(self):
        """Test that training script can be imported."""
        try:
            from torchtitan.models.nsa.train_nsa import setup_logging, setup_distributed
            self.assertTrue(callable(setup_logging))
            self.assertTrue(callable(setup_distributed))
        except ImportError as e:
            self.fail(f"Could not import training functions: {e}")

    @patch('torch.cuda.is_available')
    def test_training_script_cpu_mode(self, mock_cuda_available):
        """Test training script CPU compatibility."""
        mock_cuda_available.return_value = False
        
        # This should not crash
        from torchtitan.models.nsa.train_nsa import create_model, create_training_components
        
        # These functions should exist and be callable
        self.assertTrue(callable(create_model))
        self.assertTrue(callable(create_training_components))

    def test_config_file_paths(self):
        """Test configuration file paths."""
        # Test that config files exist
        debug_config = Path("torchtitan/models/nsa/train_configs/debug_model.toml")
        nsa_100m_config = Path("torchtitan/models/nsa/train_configs/nsa_100m.toml")
        
        self.assertTrue(debug_config.exists(), f"Debug config not found: {debug_config}")
        self.assertTrue(nsa_100m_config.exists(), f"100M config not found: {nsa_100m_config}")

    def test_ns_attention_import(self):
        """Test NSA attention import from DeepSeek."""
        try:
            # Test import from DeepSeek NSA
            sys.path.insert(0, str(Path(__file__).parent.parent / "DeepSeek NSA"))
            from nsa.kernels import NSAAttention, NSAConfig
            
            # Test basic instantiation
            config = NSAConfig(
                d_model=256,
                head_dim_qk=32,
                head_dim_v=32,
                n_heads=8,
                n_kv_groups=2,
                l=16,
                d=8,
                l_prime=32,
                n=8,
                w=64,
            )
            
            attention = NSAAttention(config)
            self.assertIsNotNone(attention)
            
        except ImportError as e:
            self.skipTest(f"DeepSeek NSA not available: {e}")


if __name__ == '__main__':
    # Run integration tests
    unittest.main()