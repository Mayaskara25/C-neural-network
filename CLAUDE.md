# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A from-scratch neural network library in C. Supports fully-connected (dense) layers, mini-batch SGD training, and binary model save/load. No build system — compiled directly with gcc.

## Build Commands

```bash
make trainer_test   # build one test
make all            # build all tests
make clean          # remove all built binaries
```

Each test target automatically compiles against all `src/*.c` files. The `compile_flags.txt` has `-Iinclude -Wall` for editor tooling (clangd/zed).

There are no automated tests, linter, or CI. The tests/ files are manual integration tests — some require stdin input (matrix_test, layer_test, model_test, backprop_test), while trainer_test is self-contained.

## Memory Ownership Rules

This is the most critical thing to get right when modifying this code:

- **Every matrix-returning function allocates**: `create_mat`, `copy_mat`, `add_mat`, `sub_mat`, `mul_mat`, `scalarmul_mat`, `transpose_mat`, `elemul_mat`, `other_op_mat`, `mse_derivative`, `mae_derivative`, `huber_derivative`, `binary_cross_entropy_derivative`, `predict` — all return matrices the **caller must `free_mat()`**.
- **Layer owns its internal matrices**: `forward_pass` and `backward_pass` free and re-allocate `layer->input`, `layer->output`, `layer->z`, `layer->delta`, `layer->dW`, `layer->db`, `layer->dx` each call. When reassigning these, free the old value first.
- **Returned pointers into layers are borrowed**: `forward_network` returns `const*` to the last layer's output. The caller does NOT own it — do not free it. It becomes invalid when the layer is freed or the next forward pass runs.
- **Ownership transfers**: `add_layer` — network takes ownership of the layer. `create_trained_model` — takes ownership of the network. Do not free these separately after handing them off.
- **Dataset deep-copies on `add_example`**: The caller can free the original matrices after adding. `Batch.inputs`/`Batch.outputs` arrays are shallow pointers into Dataset data — `free_batch()` frees only the pointer arrays, not the matrix data.

## Architecture

The dependency chain flows bottom-up:

```
matrix.h  →  layer.h  →  model.h  →  trainer.h
                                    ↗
activation.h ──────────────────────/
loss.h ───────────────────────────/
dataset.h → dataloader.h ───────/
```

- **matrix** (`src/matrix.c`): Core data type. Row-major flat `float*` array. Use `mat_at(m, r, c)` macro for element access.
- **layer** (`src/layer.c`): Single dense layer with forward/backward pass. Caches intermediate values (input, z, output) for backprop.
- **model** (`src/model.c`): Sequential container of DenseLayer pointers with dynamic array growth. Provides per-example SGD (`update_weights`) and batch SGD (`update_weights_batch`).
- **trainer** (`src/trainer.c`): High-level training loop, normalization, predict, and binary save/load. Wraps NeuralNetwork in TrainedModel which stores normalization params.
- **dataset/dataloader**: Dataset stores deep-copied examples with dynamic capacity. DataLoader provides shuffled mini-batch iteration over a Dataset.

## Normalization

`normalize_dataset` uses per-feature max-abs scaling. Each input/output feature is independently normalized to [-1, 1] by its own max absolute value. The per-feature max vectors are stored as `matrix` fields (`input_max`, `output_max`) inside `TrainedModel`, and are saved/loaded with the model binary.
