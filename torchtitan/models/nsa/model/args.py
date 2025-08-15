# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

"""
Model configuration arguments for NSA (Native Sparse Attention) models.

This module extends the Llama3 model arguments to support NSA-specific parameters
required for Native Sparse Attention computation. These parameters control the
sparse attention patterns and computational efficiency of the NSA mechanism.

NSA Parameters:
    - compression_block_size: Size of blocks used for attention compression
    - compression_stride: Stride for compression operation
    - selection_block_size: Size of blocks for attention selection
    - num_selected_blocks: Number of blocks to select for attention
    - sliding_window_size: Size of sliding window for local attention

The parameters are designed to provide fine-grained control over the trade-off
between computational efficiency and model quality in sparse attention computation.
"""

from torchtitan.models.llama3.model.args import TransformerModelArgs as Llama3TransformerModelArgs


class TransformerModelArgs(Llama3TransformerModelArgs):
    """
    Model arguments for NSA (Native Sparse Attention) models.
    
    Extends Llama3 TransformerModelArgs to support NSA-specific parameters.
    These parameters configure the Native Sparse Attention mechanism for
    efficient attention computation in large language models.
    
    Attributes:
        compression_block_size (int): Size of compression blocks (l parameter)
            Controls how input sequences are compressed for efficient attention.
            Default: 32
            
        compression_stride (int): Stride for compression operation (d parameter)
            Determines the overlap between compression blocks.
            Default: 16
            
        selection_block_size (int): Size of selection blocks (l' parameter)
            Controls granularity of attention selection from compressed representations.
            Default: 64
            
        num_selected_blocks (int): Number of blocks to select (n parameter)
            Determines how many compressed blocks participate in attention computation.
            Default: 16
            
        sliding_window_size (int): Size of sliding window for local attention (w parameter)
            Provides local context for each position in addition to selected blocks.
            Default: 512
    """
    
    # NSA-specific parameters for sparse attention computation
    compression_block_size: int = 32  # l: compression block size
    compression_stride: int = 16  # d: compression stride
    selection_block_size: int = 64  # l': selection block size
    num_selected_blocks: int = 16  # n: total blocks to select
    sliding_window_size: int = 512  # w: sliding window size
    
    # GQA configuration inherited from Llama3:
    # n_heads: number of query attention heads
    # n_kv_heads: number of key/value attention heads (for grouped query attention)
    
    def __post_init__(self):
        """
        Validate NSA configuration parameters after initialization.
        
        Performs comprehensive validation of NSA parameters to ensure:
        1. Mathematical consistency of compression and selection parameters
        2. Compatibility with grouped query attention (GQA)
        3. Sensible bounds for sparse attention computation
        
        Raises:
            AssertionError: If any NSA parameter configuration is invalid
        """
        super().__post_init__()
        
        # Validate NSA parameter relationships
        assert self.compression_block_size % self.compression_stride == 0, \
            f"compression_block_size ({self.compression_block_size}) must be divisible by compression_stride ({self.compression_stride})"
            
        assert self.selection_block_size % self.compression_stride == 0, \
            f"selection_block_size ({self.selection_block_size}) must be divisible by compression_stride ({self.compression_stride})"
            
        assert self.sliding_window_size > 0, \
            f"sliding_window_size ({self.sliding_window_size}) must be positive"
            
        assert self.num_selected_blocks > 0, \
            f"num_selected_blocks ({self.num_selected_blocks}) must be positive"
        
        # Ensure GQA compatibility - query heads must be divisible by key/value heads
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads}) for grouped query attention"