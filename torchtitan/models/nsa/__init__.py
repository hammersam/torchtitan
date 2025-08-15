# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

"""
Native Sparse Attention (NSA) registration for TorchTitan's standard trainer.

This module wires the NSA model into TorchTitan's TrainSpec so that it can be
trained via the standard entrypoint (torchtitan/train.py) using
`--model.name nsa --model.flavor <flavor>`.
"""

from .model.model import Transformer
from .model.args import TransformerModelArgs

from torchtitan.protocols.train_spec import TrainSpec, register_train_spec
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.models.llama3.infra.parallelize import parallelize_llama as parallelize_nsa
from torchtitan.models.llama3.infra.pipeline import pipeline_llama as pipeline_nsa


# Provide a couple of ready-to-use NSA model flavors
nsa_configs = {
    # Small debug model for quick sanity checks and CI
    "debug": TransformerModelArgs(
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=1,
        vocab_size=1000,
        multiple_of=128,
        max_seq_len=64,
        compression_block_size=8,
        compression_stride=4,
        selection_block_size=16,
        num_selected_blocks=4,
        sliding_window_size=32,
    ),
    # Roughly ~100M parameter configuration used in examples/tests
    "100m": TransformerModelArgs(
        dim=512,
        n_layers=6,
        n_heads=8,
        n_kv_heads=2,
        vocab_size=32000,
        multiple_of=256,
        max_seq_len=512,
        compression_block_size=16,
        compression_stride=8,
        selection_block_size=32,
        num_selected_blocks=8,
        sliding_window_size=128,
    ),
}


register_train_spec(
    TrainSpec(
        name="nsa",
        model_cls=Transformer,
        model_args=nsa_configs,
        parallelize_fn=parallelize_nsa,  # Reuse Llama3 parallelization utilities
        pipelining_fn=pipeline_nsa,  # Enable Pipeline Parallel via Llama3 infra
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=None,
    )
)

__all__ = ["Transformer", "TransformerModelArgs"]
