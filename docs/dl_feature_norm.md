# LayerNorm / BatchNorm Concepts for DL Features

When extending factor research with deep learning, normalization choices matter:

- BatchNorm:
  - Normalizes using mini-batch statistics.
  - Works well with large stable batches.
  - Can be sensitive when time-series batches are small or non-stationary.

- LayerNorm:
  - Normalizes within each sample across feature dimension.
  - Batch-size agnostic, often more stable for sequence models and small batches.

Practical guidance for quant panels:
- If training uses variable/small batch sizes or sequence models, prefer LayerNorm.
- If batch distribution is stable and large, BatchNorm can improve optimization speed.
- Keep train/validation split strictly time-aware to avoid temporal leakage.
