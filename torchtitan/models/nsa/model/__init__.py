# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

"""
NSA model components for torchtitan integration.

This module contains the core model implementations for Native Sparse Attention (NSA)
including the transformer architecture, attention mechanisms, and supporting components.

Modules:
    - model.py: Complete transformer model with NSA attention
    - args.py: Model configuration arguments and parameters

The model architecture is based on Llama3 with NSA attention replacing the standard
multi-head attention mechanism. This provides significant computational efficiency
improvements while maintaining model quality.

Key components:
    - Transformer: Main model class with full NSA integration
    - Attention: NSA-based multi-head attention mechanism
    - TransformerBlock: Individual transformer layer with NSA attention
    - RMSNorm: Root mean square normalization
    - FeedForward: Feed-forward network with SwiGLU activation
"""

from .model import Transformer, Attention, TransformerBlock, RMSNorm, FeedForward
from .args import TransformerModelArgs

__all__ = [
    "Transformer",
    "Attention", 
    "TransformerBlock",
    "RMSNorm",
    "FeedForward",
    "TransformerModelArgs"
]