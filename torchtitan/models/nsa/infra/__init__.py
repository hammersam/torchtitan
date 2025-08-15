# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Infrastructure components for NSA model distributed training.

This module provides the distributed training infrastructure for Native Sparse
Attention models within the torchtitan framework. It handles model parallelism,
FSDP sharding, and other distributed training aspects.

Key components:
    - parallelize.py: Distributed training setup and FSDP configuration
    - Model sharding strategies for efficient training
    - GPU-specific optimizations for A100/H100

The infrastructure is designed to work seamlessly with torchtitan's distributed
training capabilities, including FSDP, tensor parallelism, and pipeline parallelism.
"""

from .parallelize import parallelize_model

__all__ = ["parallelize_model"]