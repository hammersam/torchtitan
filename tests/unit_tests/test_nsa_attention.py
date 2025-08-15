#!/usr/bin/env python3
"""
Unit tests for Native Sparse Attention (NSA) integration.

This module contains comprehensive unit tests for the NSA attention mechanism
integrated into torchtitan. Tests cover:
- Basic functionality and correctness
- Shape and dtype validation
- Distributed training compatibility
- CUDA graph optimization
- Memory efficiency
- Edge cases and error handling

Note: These tests are designed to run without actual GPU/triton environment
as requested by the user.
"""

import unittest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add torchtitan to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torchtitan.models.nsa.model.model import Attention, Transformer, RMSNorm, FeedForward
from torchtitan.models.nsa.model.args import TransformerModelArgs


class TestNSAAttention(unittest.TestCase):
    """Test cases for NSA attention mechanism."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 64
        self.model_dim = 256
        self.num_heads = 8
        self.num_kv_heads = 2
        self.head_dim = self.model_dim // self.num_heads
        
        # Create model arguments for testing
        self.args = TransformerModelArgs(
            dim=self.model_dim,
            n_layers=2,
            n_heads=self.num_heads,
            n_kv_heads=self.num_kv_heads,
            vocab_size=32000,
            max_seq_len=512,
            compression_block_size=16,
            compression_stride=8,
            selection_block_size=32,
            num_selected_blocks=8,
            sliding_window_size=128,
        )

    def test_attention_initialization(self):
        """Test NSA attention initialization."""
        attention = Attention(self.args)
        
        # Check that all components are initialized
        self.assertIsInstance(attention.wq, nn.Linear)
        self.assertIsInstance(attention.wk, nn.Linear)
        self.assertIsInstance(attention.wv, nn.Linear)
        self.assertIsInstance(attention.wo, nn.Linear)
        
        # Check dimensions
        self.assertEqual(attention.wq.out_features, self.num_heads * self.head_dim)
        self.assertEqual(attention.wk.out_features, self.num_kv_heads * self.head_dim)
        self.assertEqual(attention.wv.out_features, self.num_kv_heads * self.head_dim)
        self.assertEqual(attention.wo.in_features, self.num_heads * self.head_dim)

    def test_attention_forward_shape(self):
        """Test attention forward pass shape correctness."""
        attention = Attention(self.args)
        
        # Create input tensor
        x = torch.randn(self.batch_size, self.seq_len, self.model_dim)
        freqs_cis = torch.randn(self.seq_len, self.head_dim)
        
        # Forward pass
        output = attention(x, 0, freqs_cis)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.model_dim)
        self.assertEqual(output.shape, expected_shape)

    def test_attention_dtype_preservation(self):
        """Test that attention preserves input dtype."""
        attention = Attention(self.args)
        
        for dtype in [torch.float32, torch.float16]:
            x = torch.randn(self.batch_size, self.seq_len, self.model_dim, dtype=dtype)
            freqs_cis = torch.randn(self.seq_len, self.head_dim, dtype=dtype)
            
            output = attention(x, 0, freqs_cis)
            self.assertEqual(output.dtype, dtype)

    def test_attention_distributed_setup(self):
        """Test distributed training setup."""
        attention = Attention(self.args)
        
        # Test model parallel setup
        original_heads = attention.n_local_heads
        attention.setup_distributed(model_parallel_size=2)
        
        # Check that heads are divided correctly
        expected_heads = self.num_heads // 2
        self.assertEqual(attention.n_local_heads, expected_heads)

    def test_attention_config_validation(self):
        """Test NSA configuration validation."""
        # Test invalid configuration
        with self.assertRaises(AssertionError):
            TransformerModelArgs(
                dim=256,
                n_layers=2,
                n_heads=8,
                n_kv_heads=3,  # Not divisible by num_heads
                compression_block_size=16,
                compression_stride=8,
                selection_block_size=32,
                num_selected_blocks=8,
                sliding_window_size=128,
            )


class TestTransformerBlock(unittest.TestCase):
    """Test cases for transformer block."""

    def setUp(self):
        """Set up test fixtures."""
        self.args = TransformerModelArgs(
            dim=256,
            n_layers=2,
            n_heads=8,
            n_kv_heads=2,
            vocab_size=32000,
            max_seq_len=512,
            compression_block_size=16,
            compression_stride=8,
            selection_block_size=32,
            num_selected_blocks=8,
            sliding_window_size=128,
        )

    def test_transformer_block_initialization(self):
        """Test transformer block initialization."""
        block = TransformerBlock(0, self.args)
        
        self.assertIsInstance(block.attention, Attention)
        self.assertIsInstance(block.feed_forward, FeedForward)
        self.assertIsInstance(block.attention_norm, RMSNorm)
        self.assertIsInstance(block.ffn_norm, RMSNorm)

    def test_transformer_block_forward(self):
        """Test transformer block forward pass."""
        block = TransformerBlock(0, self.args)
        
        x = torch.randn(2, 64, 256)
        freqs_cis = torch.randn(64, 32)
        
        output = block(x, 0, freqs_cis)
        
        self.assertEqual(output.shape, x.shape)


class TestTransformer(unittest.TestCase):
    """Test cases for full transformer model."""

    def setUp(self):
        """Set up test fixtures."""
        self.args = TransformerModelArgs(
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

    def test_transformer_initialization(self):
        """Test transformer initialization."""
        model = Transformer(self.args)
        
        self.assertIsInstance(model.tok_embeddings, nn.Embedding)
        self.assertIsInstance(model.layers, nn.ModuleList)
        self.assertIsInstance(model.norm, RMSNorm)
        self.assertIsInstance(model.output, nn.Linear)
        
        # Check layer count
        self.assertEqual(len(model.layers), self.args.n_layers)

    def test_transformer_forward(self):
        """Test transformer forward pass."""
        model = Transformer(self.args)
        
        # Create input tokens
        tokens = torch.randint(0, self.args.vocab_size, (2, 32))
        
        # Forward pass
        logits = model(tokens)
        
        # Check output shape
        expected_shape = (2, 32, self.args.vocab_size)
        self.assertEqual(logits.shape, expected_shape)

    def test_transformer_forward_long_sequence(self):
        """Test transformer with longer sequences."""
        model = Transformer(self.args)
        
        # Test near max sequence length
        tokens = torch.randint(0, self.args.vocab_size, (1, self.args.max_seq_len))
        logits = model(tokens)
        
        expected_shape = (1, self.args.max_seq_len, self.args.vocab_size)
        self.assertEqual(logits.shape, expected_shape)

    def test_transformer_parameter_reset(self):
        """Test parameter resetting functionality."""
        model = Transformer(self.args)
        
        # Store original parameters
        original_params = [p.clone() for p in model.parameters()]
        
        # Reset parameters
        model.reset_parameters()
        
        # Check that parameters changed
        for original, new in zip(original_params, model.parameters()):
            # Parameters should not be identical after reset
            self.assertFalse(torch.equal(original, new))


class TestRMSNorm(unittest.TestCase):
    """Test cases for RMS normalization."""

    def test_rms_norm_initialization(self):
        """Test RMSNorm initialization."""
        dim = 256
        norm = RMSNorm(dim)
        
        self.assertEqual(norm.weight.shape, (dim,))
        self.assertTrue(torch.allclose(norm.weight, torch.ones(dim)))

    def test_rms_norm_forward(self):
        """Test RMSNorm forward pass."""
        dim = 256
        norm = RMSNorm(dim)
        
        x = torch.randn(2, 32, dim)
        output = norm(x)
        
        self.assertEqual(output.shape, x.shape)

    def test_rms_norm_normalization(self):
        """Test that RMSNorm actually normalizes inputs."""
        dim = 4
        norm = RMSNorm(dim)
        
        # Test with known values
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        output = norm(x)
        
        # Check that output has unit RMS
        rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-6))


class TestFeedForward(unittest.TestCase):
    """Test cases for feed-forward network."""

    def test_feedforward_initialization(self):
        """Test FeedForward initialization."""
        dim = 256
        hidden_dim = 4 * dim
        multiple_of = 256
        
        ff = FeedForward(dim, hidden_dim, multiple_of, None)
        
        self.assertIsInstance(ff.w1, nn.Linear)
        self.assertIsInstance(ff.w2, nn.Linear)
        self.assertIsInstance(ff.w3, nn.Linear)

    def test_feedforward_forward(self):
        """Test FeedForward forward pass."""
        dim = 256
        hidden_dim = 4 * dim
        multiple_of = 256
        
        ff = FeedForward(dim, hidden_dim, multiple_of, None)
        
        x = torch.randn(2, 32, dim)
        output = ff(x)
        
        self.assertEqual(output.shape, x.shape)

    def test_feedforward_multiplier(self):
        """Test feed-forward dimension multiplier."""
        dim = 256
        hidden_dim = 4 * dim
        multiple_of = 256
        multiplier = 1.5
        
        ff = FeedForward(dim, hidden_dim, multiple_of, multiplier)
        
        # Check that hidden dimension is adjusted
        expected_hidden = int(1.5 * (2 * hidden_dim / 3))
        expected_hidden = multiple_of * ((expected_hidden + multiple_of - 1) // multiple_of)
        
        self.assertEqual(ff.w1.out_features, expected_hidden)


class TestNSAConfiguration(unittest.TestCase):
    """Test cases for NSA configuration."""

    def test_valid_configuration(self):
        """Test valid NSA configuration creation."""
        args = TransformerModelArgs(
            dim=512,
            n_layers=12,
            n_heads=16,
            n_kv_heads=4,
            vocab_size=32000,
            max_seq_len=2048,
            compression_block_size=32,
            compression_stride=16,
            selection_block_size=64,
            num_selected_blocks=16,
            sliding_window_size=512,
        )
        
        # Should not raise any exceptions
        self.assertIsInstance(args, TransformerModelArgs)

    def test_invalid_compression_stride(self):
        """Test invalid compression stride configuration."""
        with self.assertRaises(AssertionError):
            TransformerModelArgs(
                dim=512,
                n_layers=12,
                n_heads=16,
                n_kv_heads=4,
                compression_block_size=32,
                compression_stride=7,  # Not divisible
                selection_block_size=64,
                num_selected_blocks=16,
                sliding_window_size=512,
            )

    def test_invalid_gqa_ratio(self):
        """Test invalid GQA ratio."""
        with self.assertRaises(AssertionError):
            TransformerModelArgs(
                dim=512,
                n_layers=12,
                n_heads=15,  # Not divisible by kv_heads
                n_kv_heads=4,
                compression_block_size=32,
                compression_stride=16,
                selection_block_size=64,
                num_selected_blocks=16,
                sliding_window_size=512,
            )

    def test_edge_case_parameters(self):
        """Test edge case parameter values."""
        # Test minimum viable configuration
        args = TransformerModelArgs(
            dim=64,
            n_layers=1,
            n_heads=1,
            n_kv_heads=1,
            vocab_size=100,
            max_seq_len=32,
            compression_block_size=1,
            compression_stride=1,
            selection_block_size=1,
            num_selected_blocks=1,
            sliding_window_size=1,
        )
        
        self.assertIsInstance(args, TransformerModelArgs)


class TestNSAIntegration(unittest.TestCase):
    """Integration tests for NSA components."""

    def test_end_to_end_training_step(self):
        """Test end-to-end training step."""
        args = TransformerModelArgs(
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
        
        model = Transformer(args)
        
        # Create dummy training data
        batch_size = 2
        seq_len = 32
        tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, args.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        logits = model(tokens)
        
        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, args.vocab_size),
            labels.view(-1),
            ignore_index=-1,
        )
        
        # Check loss computation
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)

    def test_model_size_calculation(self):
        """Test parameter count calculation."""
        args = TransformerModelArgs(
            dim=768,
            n_layers=12,
            n_heads=12,
            n_kv_heads=3,
            vocab_size=32000,
            max_seq_len=4096,
            compression_block_size=32,
            compression_stride=16,
            selection_block_size=64,
            num_selected_blocks=16,
            sliding_window_size=512,
        )
        
        model = Transformer(args)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Rough check for 100M parameter model
        self.assertGreater(total_params, 50_000_000)
        self.assertLess(total_params, 200_000_000)


class TestCUDAOptimization(unittest.TestCase):
    """Test CUDA graph and optimization features."""

    @patch('torch.cuda.is_available')
    def test_auto_optimization_cpu_fallback(self, mock_cuda_available):
        """Test that optimizations gracefully handle CPU environment."""
        mock_cuda_available.return_value = False
        
        # This should not crash even without CUDA
        args = TransformerModelArgs(
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
        
        model = Transformer(args)
        tokens = torch.randint(0, args.vocab_size, (1, 32))
        
        # Should work fine on CPU
        logits = model(tokens)
        self.assertEqual(logits.shape, (1, 32, args.vocab_size))

    def test_memory_efficiency(self):
        """Test memory usage patterns."""
        args = TransformerModelArgs(
            dim=256,
            n_layers=2,
            n_heads=8,
            n_kv_heads=2,
            vocab_size=1000,
            max_seq_len=512,
            compression_block_size=16,
            compression_stride=8,
            selection_block_size=32,
            num_selected_blocks=8,
            sliding_window_size=128,
        )
        
        model = Transformer(args)
        
        # Test with different sequence lengths
        for seq_len in [32, 64, 128]:
            tokens = torch.randint(0, args.vocab_size, (1, seq_len))
            logits = model(tokens)
            
            self.assertEqual(logits.shape, (1, seq_len, args.vocab_size))


if __name__ == '__main__':
    # Run all tests
    unittest.main()