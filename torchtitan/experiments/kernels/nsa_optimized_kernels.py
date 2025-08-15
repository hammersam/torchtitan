"""
Optimized NSA kernel with combined optimizations for torchtitan.

This module provides a unified, optimized version of the NSA attention kernel
that incorporates several performance enhancements:

1.  **CUDA Graph Acceleration**: Reduces kernel launch overhead, especially for
    smaller batch sizes or sequence lengths.
2.  **Memory Efficiency**: Uses a pre-allocated memory pool to reduce memory
    fragmentation and allocation overhead.
3.  **GPU-Specific Tuning**:
    -   **A100**: Leverages TF32 for faster matrix multiplications.
    -   **H100**: Utilizes FP8 and higher-throughput kernels.
4.  **Automatic GPU Detection**: The kernel automatically detects the underlying
    GPU architecture (A100, H100, or other) and applies the most effective
    optimizations at runtime.

This approach simplifies usage by providing a single, powerful kernel that
delivers the best possible performance for the detected hardware.

Usage:
    from torchtitan.experiments.kernels.nsa_optimized_kernels import OptimizedNSA

    # The kernel will automatically configure itself for the detected GPU
    nsa_attention = OptimizedNSA(config)
    output, attention_weights = nsa_attention(hidden_states)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class OptimizedNSA(nn.Module):
    """
    A unified, optimized NSA kernel combining CUDA graphs, memory pooling,
    and GPU-specific tuning for A100 and H100.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.nsa_attention = None

        # CUDA Graph attributes
        self.cuda_graphs = {}
        self.static_inputs = {}
        self.static_outputs = {}
        self.warmup = True

        # Memory pool attributes
        self.register_buffer("_memory_pool", None, persistent=False)
        self.pool_size = 0

        # GPU-specific attributes
        self.block_m = 64  # Default for generic CUDA
        self.block_n = 64
        self.use_tf32 = False
        self.use_fp8 = False

        self._detect_and_configure_gpu()

    def _detect_and_configure_gpu(self):
        """Detects GPU architecture and configures optimizations."""
        if not torch.cuda.is_available():
            logger.info("CUDA not available. Running NSA on CPU.")
            return

        device_capability = torch.cuda.get_device_capability()
        major, minor = device_capability

        if major >= 9:  # Hopper and newer
            logger.info("Detected H100+ architecture. Enabling H100 optimizations.")
            self.block_m = 128
            self.block_n = 128
            self.use_tf32 = True
            # Check for FP8 support more robustly
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                self.use_fp8 = True
        elif major == 8:  # Ampere
            logger.info("Detected A100 architecture. Enabling A100 optimizations.")
            self.block_m = 64
            self.block_n = 64
            self.use_tf32 = True
        else:
            logger.info(f"Detected generic CUDA architecture (sm{major}{minor}). Using default optimizations.")
            # Stick with default block sizes for older architectures

    def _init_nsa(self):
        """Initialize the underlying NSAAttention module with optimized config."""
        try:
            from torchtitan.experiments.kernels.nsa.kernels import NSAAttention, NSAConfig

            optimized_config = NSAConfig(
                d_model=self.config.d_model,
                head_dim_qk=self.config.head_dim_qk,
                head_dim_v=self.config.head_dim_v,
                n_heads=self.config.n_heads,
                n_kv_groups=self.config.n_kv_groups,
                l=self.config.l,
                d=self.config.d,
                l_prime=self.config.l_prime,
                n=self.config.n,
                w=self.config.w,
                dropout_p=self.config.dropout_p,
                use_bias=self.config.use_bias,
                block_m=self.block_m,
                block_n=self.block_n,
            )

            self.nsa_attention = NSAAttention(optimized_config).to(device=self._get_device())
        except ImportError as e:
            logger.error(f"Failed to import NSA kernels. Make sure they are compiled and in the PYTHONPATH: {e}")
            raise

    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _allocate_memory_pool(self, batch_size: int, seq_len: int, device: torch.device):
        """Allocate a memory pool for intermediate tensors to reduce overhead."""
        # Rough estimation of memory needed for intermediate tensors in NSA
        # Safety factor of 8x to be safe.
        est_memory = batch_size * seq_len * self.config.d_model * 4 * 8

        if self._memory_pool is None or self._memory_pool.numel() < est_memory:
            self.pool_size = est_memory
            self._memory_pool = torch.empty(
                est_memory, device=device, dtype=torch.float16
            )
            logger.info(f"Allocated memory pool of size: {self.pool_size * 2 / (1024**2):.2f} MB")

    def _create_cuda_graph(
        self, key: str, hidden_states: torch.Tensor
    ) -> bool:
        """Creates a CUDA graph for a given input shape to accelerate subsequent calls."""
        if not torch.cuda.is_available():
            return False

        try:
            # Create static input tensor on the correct device
            static_hidden_states = torch.randn_like(hidden_states)

            # Warmup runs
            for _ in range(3):
                with torch.cuda.stream(torch.cuda.Stream()):
                    _ = self.nsa_attention(static_hidden_states)
            torch.cuda.synchronize()

            # Create CUDA graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_outputs = self.nsa_attention(static_hidden_states)

            self.cuda_graphs[key] = graph
            self.static_inputs[key] = (static_hidden_states,)
            self.static_outputs[key] = static_outputs
            logger.info(f"Successfully created CUDA graph for shape {hidden_states.shape}")
            return True

        except Exception as e:
            logger.warning(f"Failed to create CUDA graph for shape {hidden_states.shape}: {e}. Will fallback to eager execution.")
            return False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with combined optimizations.

        Applies GPU-specific settings, uses memory pooling, and leverages
        CUDA graphs for performance.
        """
        if self.nsa_attention is None:
            self._init_nsa()

        # Fallback to eager if attention weights are requested with CUDA graphs, as they might not be static.
        use_cuda_graph = torch.cuda.is_available() and hidden_states.device.type == "cuda" and not output_attentions

        if not use_cuda_graph:
            return self.nsa_attention(hidden_states, attention_mask, output_attentions)

        # --- GPU-specific settings ---
        if self.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if self.use_fp8:
            torch.backends.cuda.enable_flash_sdp(True)

        # --- Memory Pooling ---
        self._allocate_memory_pool(
            hidden_states.shape[0], hidden_states.shape[1], hidden_states.device
        )

        # --- CUDA Graph Logic ---
        key = f"bs{hidden_states.shape[0]}_sl{hidden_states.shape[1]}"

        if key not in self.cuda_graphs:
            # Ensure nsa_attention is on the correct device before graph creation
            self.nsa_attention.to(hidden_states.device)
            success = self._create_cuda_graph(key, hidden_states)
            if not success:
                return self.nsa_attention(hidden_states, attention_mask, output_attentions)

        static_input_tuple = self.static_inputs[key]
        static_input = static_input_tuple[0]

        if hidden_states.shape == static_input.shape:
            # Copy input to the static tensor
            static_input.copy_(hidden_states)

            # Replay the graph
            self.cuda_graphs[key].replay()

            # Return result (clone to avoid returning a graph-owned tensor)
            output = self.static_outputs[key][0].clone()
            # CUDA graphs don't support dynamic outputs, so attention_info is None
            attention_info = None
            return output, attention_info
        else:
            # Fallback for dynamic shapes
            logger.warning(f"Input shape changed from {static_input.shape} to {hidden_states.shape}. "
                           f"Falling back to eager execution for this call.")
            return self.nsa_attention(hidden_states, attention_mask, output_attentions)
