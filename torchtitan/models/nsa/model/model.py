# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Complete transformer model implementation with Native Sparse Attention (NSA).

This module provides the full transformer architecture with NSA attention mechanism
integrated into the torchtitan framework. The implementation is based on Llama3
architecture but replaces the standard multi-head attention with Native Sparse
Attention for improved computational efficiency.

Key Features:
- Native Sparse Attention with triton kernel acceleration
- Rotary position embeddings (RoPE)
- Grouped Query Attention (GQA) support
- RMSNorm for layer normalization
- SwiGLU activation in feed-forward networks
- Full compatibility with torchtitan distributed training
- CUDA graph optimization support for A100/H100 GPUs

Architecture Overview:
    1. Token embeddings: Vocabulary to dense representations
    2. Transformer blocks: Stack of N layers with NSA attention
    3. RMSNorm: Final layer normalization
    4. Output projection: Dense to vocabulary

Each transformer block contains:
    - NSA attention with compression, selection, and sliding window
    - Feed-forward network with SwiGLU activation
    - RMSNorm for both attention and feed-forward paths
    - Residual connections throughout

Usage:
    from torchtitan.models.nsa.model import Transformer, TransformerModelArgs
    
    # Model configuration
    args = TransformerModelArgs(
        dim=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=3,  # GQA with 4:1 ratio
        # ... other parameters
    )
    
    # Initialize model
    model = Transformer(args)
    
    # Forward pass
    logits = model(tokens)  # tokens: [batch_size, seq_len]
"""

import torch
import torch.nn.functional as F
from torch import nn
import sys
from pathlib import Path

# Add DeepSeek NSA path to sys.path for kernel imports
nsa_path = str(Path(__file__).parent.parent.parent.parent / "DeepSeek NSA")
if nsa_path not in sys.path:
    sys.path.insert(0, nsa_path)

from torchtitan.experiments.kernels.nsa.kernels import NSAAttention, NSAConfig
from torchtitan.models.attention import init_attention_mask
from torchtitan.protocols.train_spec import ModelProtocol

from .args import TransformerModelArgs


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float | None): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Multi-head attention with Native Sparse Attention (NSA) support.
    
    This class implements the attention mechanism using Native Sparse Attention,
    which provides efficient sparse attention computation through compression,
    selection, and sliding window operations. The implementation supports both
    single-GPU and distributed training configurations.
    
    Key features:
    - Native Sparse Attention with triton kernel acceleration
    - Rotary position embeddings (RoPE) support
    - Grouped Query Attention (GQA) with configurable head ratios
    - Automatic optimization selection for different GPU architectures
    - FSDP-compatible tensor sharding
    
    Architecture:
        1. Linear projections for Q, K, V transformations
        2. Rotary position embeddings application
        3. Native Sparse Attention computation
        4. Output projection back to model dimension
    
    Args:
        args: TransformerModelArgs containing model configuration
    """

    def __init__(self, args: TransformerModelArgs):
        super().__init__()
        self.args = args
        self.n_kv_heads = args.n_kv_heads
        
        # Initialize head counts for distributed training
        # These will be updated during FSDP setup based on model parallelism
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # Create NSA configuration for sparse attention computation
        # Maps model parameters to NSA kernel parameters
        self.nsa_config = NSAConfig(
            d_model=args.dim,  # Model dimension
            head_dim_qk=self.head_dim,  # Query/key head dimension
            head_dim_v=self.head_dim,  # Value head dimension
            n_heads=args.n_heads,  # Total attention heads
            n_kv_groups=args.n_kv_heads,  # Key/value groups for GQA
            l=args.compression_block_size,  # Compression block size
            d=args.compression_stride,  # Compression stride
            l_prime=args.selection_block_size,  # Selection block size
            n=args.num_selected_blocks,  # Number of selected blocks
            w=args.sliding_window_size,  # Sliding window size for local attention
            dropout_p=args.dropout,  # Dropout probability
            use_bias=False,  # No bias in linear layers
        )

        # Initialize NSA attention with automatic optimization
        # Uses OptimizedNSA for GPU-specific optimizations (A100/H100)
        try:
            from torchtitan.experiments.kernels.nsa_optimized_kernels import OptimizedNSA
            self.nsa_attention = OptimizedNSA(self.nsa_config)
        except ImportError:
            # Fallback to standard NSA attention if optimized kernels not available
            self.nsa_attention = NSAAttention(self.nsa_config)

        # Linear projections for Q, K, V, and output transformations
        # These will be automatically sharded by FSDP during distributed setup
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.wq.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wk.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wv.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        
    def setup_distributed(self, model_parallel_size: int = 1):
        """
        Setup for distributed training with model parallelism.
        
        Adjusts head counts based on model parallelism degree. This ensures
        correct attention computation when the model is sharded across GPUs.
        
        Args:
            model_parallel_size: Number of model parallel workers
        """
        self.n_local_heads = self.args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for NSA attention computation.
        
        Computes multi-head attention using Native Sparse Attention mechanism.
        Handles the complete attention pipeline from input projection to output.
        
        Args:
            x: Input tensor [batch_size, seq_len, model_dim]
            start_pos: Starting position for rotary embeddings (used in inference)
            freqs_cis: Precomputed rotary frequency tensor
            mask: Optional attention mask (NSA handles causal masking internally)
            
        Returns:
            Attention output tensor [batch_size, seq_len, model_dim]
        """
        bsz, seqlen, _ = x.shape

        # Project input to query, key, and value representations
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape for multi-head attention: [batch, seq, heads, head_dim]
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Apply rotary position embeddings to queries and keys
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Compute Native Sparse Attention
        # NSA handles key/value repetition internally for GQA
        output = self.nsa_attention(xq, xk, xv)

        # Reshape and project back to model dimension
        output = output.view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    Feed-forward network with SwiGLU activation.
    
    Implements the feed-forward component of transformer layers using SwiGLU
    activation for improved performance over standard ReLU or GELU activations.
    The network uses a gated linear unit approach with three linear projections.
    
    Architecture:
        1. First projection: Expand dimension to hidden size
        2. Second projection: Gating mechanism with SwiGLU
        3. Third projection: Compress back to model dimension
    
    Mathematical formulation:
        output = W2 * (Swish(W1(x)) * W3(x))
    
    Args:
        dim: Input and output dimension
        hidden_dim: Target hidden dimension (will be adjusted based on multiplier)
        multiple_of: Ensure hidden dimension is multiple of this value
        ffn_dim_multiplier: Optional multiplier for hidden dimension
    """

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: float | None):
        super().__init__()
        
        # Adjust hidden dimension based on multiplier
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        
        # Ensure hidden dimension is multiple of specified value for efficiency
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # Three linear projections for SwiGLU activation
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Up projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Gating projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor [batch_size, seq_len, model_dim]
            
        Returns:
            Output tensor [batch_size, seq_len, model_dim]
        """
        # SwiGLU activation: W2 * (Swish(W1(x)) * W3(x))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    """
    Single transformer layer with NSA attention and feed-forward network.
    
    This class represents one layer of the transformer architecture, consisting of:
    1. NSA attention mechanism with pre-layer normalization
    2. Feed-forward network with pre-layer normalization
    3. Residual connections around both components
    
    The layer follows the pre-norm transformer architecture where normalization
    is applied before each sub-layer rather than after.
    
    Args:
        layer_id: Unique identifier for this layer (used for debugging/tracking)
        args: TransformerModelArgs containing model configuration
    """

    def __init__(self, layer_id: int, args: TransformerModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.layer_id = layer_id
        
        # NSA attention mechanism with pre-layer normalization
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # Feed-forward network with pre-layer normalization
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,  # Standard 4x expansion
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer layer.
        
        Implements the pre-norm transformer architecture with residual connections:
        1. Layer normalization before attention
        2. NSA attention with residual connection
        3. Layer normalization before feed-forward
        4. Feed-forward network with residual connection
        
        Args:
            x: Input tensor [batch_size, seq_len, model_dim]
            start_pos: Starting position for rotary embeddings
            freqs_cis: Precomputed rotary frequency tensor
            mask: Optional attention mask (handled by NSA internally)
            
        Returns:
            Output tensor [batch_size, seq_len, model_dim]
        """
        # Attention sub-layer with pre-norm and residual connection
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        
        # Feed-forward sub-layer with pre-norm and residual connection
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, weight_init_std: float):
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.attention.init_weights(weight_init_std)
        self.feed_forward.init_weights(weight_init_std)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    RMSNorm provides efficient layer normalization by normalizing inputs based
    on their root mean square, rather than the traditional mean and variance
    approach. This reduces computational overhead while maintaining stability.
    
    Mathematical formulation:
        x_norm = x / sqrt(mean(x^2) + eps)
        output = x_norm * weight
    
    Args:
        dim: Dimension of the input tensor
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to input tensor.
        
        Args:
            x: Input tensor to normalize
            
        Returns:
            Normalized tensor
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RMS normalization.
        
        Args:
            x: Input tensor [..., dim]
            
        Returns:
            Normalized tensor [..., dim]
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Transformer(nn.Module, ModelProtocol):
    """
    Complete transformer model with Native Sparse Attention (NSA).
    
    This class implements the full transformer architecture with NSA attention
    mechanism integrated into the torchtitan framework. The model is based on
    Llama3 architecture but uses Native Sparse Attention for improved efficiency.
    
    Architecture:
        1. Token embeddings: Map vocabulary indices to dense representations
        2. Transformer blocks: Stack of N layers with NSA attention and feed-forward
        3. Final layer normalization: RMSNorm applied to output
        4. Output projection: Map back to vocabulary space
    
    Key features:
    - Native Sparse Attention for efficient attention computation
    - Rotary position embeddings (RoPE) for position encoding
    - Grouped Query Attention (GQA) support
    - RMSNorm for layer normalization
    - SwiGLU activation in feed-forward networks
    - Full torchtitan integration with distributed training support
    
    Args:
        args: TransformerModelArgs containing model configuration
    """

    def __init__(self, args: TransformerModelArgs):
        super().__init__()
        self.model_args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        # Token embeddings: Map vocabulary indices to dense representations
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        # Stack of transformer layers
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, args)

        # Final layer normalization
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # Output projection: Map back to vocabulary space
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Persistent buffer for RoPE freqs (see llama3 notes)
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        # Initialize weights once at construction
        self.init_weights()

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def init_weights(self, buffer_device: torch.device | None = None):
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)

        # Compute per-layer init std like Llama3
        for layer_id, layer in self.layers.items():
            lid = int(layer_id)
            if self.model_args.depth_init:
                weight_init_std = 0.02 / (2 * (lid + 1)) ** 0.5
            else:
                weight_init_std = 0.02 / (2 * self.model_args.n_layers) ** 0.5
            layer.init_weights(weight_init_std)

        if self.norm is not None:
            self.norm.reset_parameters()

        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def forward(
        self,
        tokens: torch.Tensor,
        eos_id: int | None = None,
        input_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Optional mask setup for certain attention types
        init_attention_mask(input_batch if input_batch is not None else tokens, eos_id=eos_id)

        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        # Training path uses start_pos=0; inference can pass non-zero via pipeline wrappers
        start_pos = 0
        for layer in self.layers.values():
            h = layer(h, start_pos, self.freqs_cis)

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Forward pass through the complete transformer model.
        
        Processes input tokens through the full transformer architecture:
        1. Token embedding lookup
        2. Rotary position embedding application
        3. Sequential processing through transformer layers
        4. Final normalization and output projection
        
        Args:
            tokens: Input token indices [batch_size, seq_len]
            start_pos: Starting position for rotary embeddings (used in inference)
            
        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        _bsz, seqlen = tokens.shape
        
        # Token embedding lookup
        h = self.tok_embeddings(tokens)
        
        # Get relevant rotary embeddings for this sequence
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # Process through all transformer layers
        # NSA handles causal masking internally, so no need for explicit mask
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask=None)
            
        # Final layer normalization
        h = self.norm(h)
        
        # Output projection to vocabulary space
        output = self.output(h)
        return output.float()

    def reset_parameters(self):
        """
        Reset all model parameters to their initial values.
        
        This method reinitializes all learnable parameters in the model,
        which is useful for training from scratch or debugging initialization
        issues.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
            elif isinstance(module, nn.Embedding):
                module.reset_parameters()
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
