# NSA Integration in TorchTitan

This document summarizes the changes made to integrate Native Sparse Attention (NSA) into TorchTitan's standard training pipeline and provides recommended usage.

## Changes Made

- TrainSpec registration for NSA
  - File: `torchtitan/models/nsa/__init__.py`
  - Registered `model.name = nsa` with two flavors:
    - `debug`: small 2-layer model for smoke tests
    - `100m`: medium configuration for examples
  - Reused common builders: optimizers, LR schedulers, tokenizer, dataloader, loss, validator.
  - Enabled parallelization via `llama3.parallelize_llama` and Pipeline Parallel via `llama3.pipeline_llama`.

- NSA model alignment to ModelProtocol
  - File: `torchtitan/models/nsa/model/model.py`
  - Implemented `init_weights()` and `forward(tokens, eos_id=None, input_batch=None)`.
  - Normalized attributes to match trainer expectations:
    - `self.model_args` holds arguments
    - `self.layers` is `nn.ModuleDict()` with keys `"0".."n-1"` (visible as `layers.0`, `layers.1`, ...)
    - Persistent buffer `freqs_cis` with precompute helper
    - Per-module weight initialization (similar to Llama3)

- Removed redundant bespoke training script
  - Deleted: `torchtitan/models/nsa/train_nsa.py`
  - All training now goes through `torchtitan/train.py`.

## Additional Cleanup

- Removed the unused NSA parallelization helper since we reuse Llama3 infra:
  - Deleted: `torchtitan/models/nsa/infra/parallelize.py`

## Recommended Usage

- Single GPU (debug flavor):
  - `python -m torchtitan.train --model.name nsa --model.flavor debug`

- Multi-GPU (e.g., 4 GPUs):
  - `torchrun --nproc_per_node=4 -m torchtitan.train --model.name nsa --model.flavor 100m`

- With a TOML config:
  - `python -m torchtitan.train --job.config_file path/to/config.toml`

Notes:
- The standard training pipeline honors `training.*`, `parallelism.*`, `checkpoint.*`, etc., from the TOML.
- Pipeline Parallel is enabled for NSA via the Llama3 pipeline infra. Configure with `parallelism.pipeline_parallel_*` options as for Llama3.

## Implementation Notes

- Parallelization and PP
  - `parallelize_llama` applies TP/CP/FSDP/HSDP consistently, leveraging `ParallelDims` and world mesh naming.
  - `pipeline_llama` performs model partitioning by module FQNs (`tok_embeddings`, `layers.i`, `norm`, `output`). NSA model exposes compatible names.

- Loss and Dataloader
  - Training uses the standard cross-entropy loss (`build_cross_entropy_loss`).
  - Dataloader uses HF dataset wrappers and tokenizer from TorchTitan components.

- Metrics and Checkpointing
  - Standard `MetricsProcessor` and `CheckpointManager` are used through the trainer; no NSA-specific code required.

## Next Steps (Optional)

- Remove `torchtitan/models/nsa/infra/parallelize.py` in a follow-up cleanup if confirmed unused across your workflows.
- Add NSA-specific state_dict adapter if you plan to export/import HF safetensors for NSA weights.
