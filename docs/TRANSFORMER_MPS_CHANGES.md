# Transformer + Apple Metal (MPS) Changes

## What changed

### 1) New transformer model

- Added `models/Transformer_w_ref.py`.
- This model keeps the same input/output interface as existing Remora models:
  - Input signal tensor: `[batch, 1, chunk_len]`
  - Input sequence tensor: `[batch, 4 * kmer_len, chunk_len]`
  - Output logits tensor: `[batch, num_out]`
- Architecture:
  - 1x1 projections for signal and sequence channels
  - Concatenation into a token embedding
  - Transformer encoder stack
  - Mean pooling + classification head

### 2) Apple Metal (MPS) device support

- Updated `src/remora/util.py` `parse_device()`:
  - Preserves legacy integer CUDA device usage (e.g. `--device 0` -> `cuda:0`)
  - Accepts `"mps"` explicitly
  - Accepts `"metal"` as an alias and maps it to `"mps"`
  - Raises a clear error when MPS is requested but unavailable
- Updated CLI argument parsing in `src/remora/parsers.py`:
  - `infer`, `infer_duplex`, and `validate` now accept string devices, not only integers.
  - Example valid values:
    - `--device 0`
    - `--device cuda:0`
    - `--device mps`
    - `--device metal`

### 3) Tests

- Added `tests/test_model_extensions.py`:
  - Verifies transformer forward-pass output shape.
  - Verifies `metal` alias behavior for MPS.
  - Verifies integer device compatibility behavior.

## How to run with MPS on Mac

### Inference

```bash
remora infer from_pod5_and_bam <pod5> <bam> --model <model.pt> --out-bam <out.bam> --device mps
```

### Validation

```bash
remora validate from_remora_dataset <dataset.cfg> --model <model.pt> --device mps
```

### Training

```bash
remora model train <dataset.cfg> --model models/Transformer_w_ref.py --device mps
```

## Performance comparison template

Fill this table as you benchmark baseline vs transformer.

| Experiment | Device | Dataset | Model | Epochs | Batch Size | Val Acc | Filtered Val Acc | Throughput (chunks/s) | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| Baseline-ConvLSTM | mps | TODO | `models/ConvLSTM_w_ref.py` | TODO | TODO | TODO | TODO | TODO | TODO |
| Baseline-Conv | mps | TODO | `models/Conv_w_ref.py` | TODO | TODO | TODO | TODO | TODO | TODO |
| Transformer-v1 | mps | TODO | `models/Transformer_w_ref.py` | TODO | TODO | TODO | TODO | TODO | TODO |

## Suggested apples-to-apples benchmarking protocol

1. Keep dataset, chunk context, and k-mer context fixed across runs.
2. Keep train/validation split settings fixed.
3. Run each model with at least 3 seeds.
4. Report mean and standard deviation for validation metrics.
5. Record wall-clock training time and inference throughput.
