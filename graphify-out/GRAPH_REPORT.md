# Graph Report - .  (2026-07-15)

## Corpus Check
- Corpus is ~9,536 words - fits in a single context window. You may not need a graph.

## Summary
- 175 nodes · 379 edges · 12 communities (11 shown, 1 thin omitted)
- Extraction: 78% EXTRACTED · 22% INFERRED · 0% AMBIGUOUS · INFERRED: 84 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Network Container & Trainer
- Matrix Core & Dense Layer
- CSV I/O & Dataset Generators
- API Reference Documentation
- Loss Functions & Gradients
- Architecture & Module Overview
- Mini-Batch DataLoader
- XOR Dataset Generator
- Build Instructions

## God Nodes (most connected - your core abstractions)
1. `train()` - 21 edges
2. `create_mat()` - 20 edges
3. `free_mat()` - 19 edges
4. `Memory Ownership Rules` - 16 edges
5. `load_model_network()` - 14 edges
6. `sub_mat()` - 11 edges
7. `predict()` - 11 edges
8. `backward_pass()` - 10 edges
9. `Architecture (Dependency Chain)` - 10 edges
10. `STRUCT DATA MEMBERS section` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Memory Ownership Rules` --semantically_similar_to--> `Memory Ownership (README explanation)`  [INFERRED] [semantically similar]
  CLAUDE.md → README.txt
- `Memory Ownership (README explanation)` --rationale_for--> `create_dense_layer()`  [EXTRACTED]
  README.txt → src/layer.c
- `Memory Ownership Rules` --references--> `mse_derivative()`  [EXTRACTED]
  CLAUDE.md → src/loss.c
- `Memory Ownership Rules` --references--> `mae_derivative()`  [EXTRACTED]
  CLAUDE.md → src/loss.c
- `Memory Ownership Rules` --references--> `huber_derivative()`  [EXTRACTED]
  CLAUDE.md → src/loss.c

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Architecture Dependency Chain (matrix -> layer -> model -> trainer, with activation/loss/dataset/dataloader feeding trainer)** — src_matrix_module, src_layer_module, src_model_module, src_trainer_module, include_activation_module, include_loss_module, src_dataset_module, src_dataloader_module [INFERRED 0.85]
- **End-to-end Training/Prediction Usage Pipeline** — src_csv_csv_to_dataset, src_model_create_neural_network, src_trainer_create_trained_model, src_trainer_normalize_dataset, src_dataloader_create_dataloader, src_trainer_train, src_trainer_predict, src_trainer_save_model, src_trainer_load_model_network [EXTRACTED 1.00]

## Communities (12 total, 1 thin omitted)

### Community 0 - "Network Container & Trainer"
Cohesion: 0.13
Nodes (33): ActivationID, FILE, TO BE IMPLEMENTED (CLI commands), USAGE PIPELINE, free_dense_layer(), accumulate_network_gradients(), add_layer(), backward_network() (+25 more)

### Community 1 - "Matrix Core & Dense Layer"
Cohesion: 0.21
Nodes (26): Memory Ownership Rules, accumulate_gradients(), backward_pass(), DenseLayer, matrix, create_dense_layer(), forward_pass(), zero_accumulators() (+18 more)

### Community 2 - "CSV I/O & Dataset Generators"
Cohesion: 0.13
Nodes (24): CSV, generate_rows(), main(), position_for(), write_csv(), generate_rows(), main(), price_for() (+16 more)

### Community 3 - "API Reference Documentation"
Cohesion: 0.12
Nodes (19): ACTIVATION FUNCTIONS section, ActivationID enum, Batch struct, CSV section, CSV struct, DATALOADER section, DataLoader struct, DATASET section (+11 more)

### Community 4 - "Loss Functions & Gradients"
Cohesion: 0.32
Nodes (14): binary_cross_entropy(), binary_cross_entropy_derivative(), LossID, matrix, get_loss_fxn(), huber(), huber_derivative(), mae() (+6 more)

### Community 5 - "Architecture & Module Overview"
Cohesion: 0.16
Nodes (14): Architecture (Dependency Chain), Build Commands, C Neural Network Library (Project Overview), Normalization (Per-Feature Max-Abs Scaling), activation module (activation.h), loss module (loss.h), TrainedModel struct, TRAINER section (+6 more)

### Community 6 - "Mini-Batch DataLoader"
Cohesion: 0.35
Nodes (10): Batch, DataLoader, Dataset, create_dataloader(), free_batch(), free_dataloader(), has_next_batch(), next_batch() (+2 more)

### Community 9 - "XOR Dataset Generator"
Cohesion: 0.83
Nodes (3): jittered_rows(), main(), write_csv()

## Knowledge Gaps
- **15 isolated node(s):** `Build Commands`, `matrix module (src/matrix.c)`, `layer module (src/layer.c)`, `model module (src/model.c)`, `activation module (activation.h)` (+10 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `csv_to_dataset()` connect `CSV I/O & Dataset Generators` to `Matrix Core & Dense Layer`, `Loss Functions & Gradients`?**
  _High betweenness centrality (0.227) - this node is a cross-community bridge._
- **Why does `free_mat()` connect `Loss Functions & Gradients` to `Network Container & Trainer`, `Matrix Core & Dense Layer`, `CSV I/O & Dataset Generators`?**
  _High betweenness centrality (0.193) - this node is a cross-community bridge._
- **Why does `Memory Ownership Rules` connect `Matrix Core & Dense Layer` to `Network Container & Trainer`, `API Reference Documentation`, `Loss Functions & Gradients`, `Architecture & Module Overview`?**
  _High betweenness centrality (0.187) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `train()` (e.g. with `free_batch()` and `has_next_batch()`) actually correct?**
  _`train()` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `create_mat()` (e.g. with `csv_to_dataset()` and `create_dense_layer()`) actually correct?**
  _`create_mat()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `free_mat()` (e.g. with `csv_to_dataset()` and `free_dataset()`) actually correct?**
  _`free_mat()` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Memory Ownership Rules` (e.g. with `C Neural Network Library (Project Overview)` and `Memory Ownership (README explanation)`) actually correct?**
  _`Memory Ownership Rules` has 2 INFERRED edges - model-reasoned connections that need verification._