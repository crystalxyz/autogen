# Overview
This folder aims to contain the scripts used to analyze the kv tensor difference across models of the same model family and train a neural network to do transformation with minimal loss. We are interested in the prefill kvcache only.

# Setup
- With each HumanEval query, feed it into qwen3-0.6b, qwen3-1.7b, qwen3-4b, and qwen3-8b models. Get the resulting prefill KV tensors.
- Analyze any similarity across KV tensors using CKA metrics
- Train a NN using the training data, and test its performance accuracy
- The NN structure could be a MLP to begin with

# Execution Plan

## Phase 1: Data Collection

**Goal:** Run each HumanEval prompt through `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`, and `qwen3-8b` and capture the prefill KV tensors.

1. Load HumanEval prompts (just the prompt/docstring, not the solution)
2. For each model, run a forward pass (prefill only — no generation needed)
3. Hook into the model's attention layers to extract K and V tensors at each layer
4. Save tensors to disk, indexed by `(model, problem_id, layer)`

```bash
python collect_kv.py \
    --models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B \
    --output_dir data/
```

**Script:** `collect_kv.py`
**Output:** Four tensor archives (one per model), shape `[num_problems, num_layers, seq_len, num_heads, head_dim]`

---

## Phase 2: CKA Similarity Analysis

Idea 1: Analyze layer similarity across models
Goal: For models with matching layer number, we can compare corresponding layers in a 1-1 mapping.
- The input K/V tensor has shape [num_layer, num_head, seq_len, head_dim]. For each layer, we flatten the tensor to be [seq_len, num_head * head_dim]. Then we compute the similarity between each layers of the corresponding model.
- The target will be 0.6B-1.7B and 4B-8B.

Idea 2: Analyze similarity between adjacent layers
Goal: To understand if it's possible to combine multiple adjacent layers and map them to one layer in a smaller model.
- If adjacent layers in the larger model have high CKA similarity, their KV representations are redundant and could potentially be merged. For each model, compute CKA between layer i and layer i+1 (and i+2, i+3) to identify clusters of similar layers. For example, if layers 10-13 in the 8b model all show CKA > 0.9 with each other, they could potentially be mapped to a single layer in the 4b model via a learned projection.
- 

Idea 3: Analyze cross-model layer correspondence
Goal: Identify which layers in a larger model correspond to which layers in a smaller model.
Details: Compute the full L_large × L_small CKA heatmap. Look for diagonal-like structure indicating ordered correspondence, or off-diagonal bright spots indicating that certain layers in the large model align with unexpected layers in the small model. This mapping informs which layer pairs are candidates for KV cache transfer.

Idea 4: Analyze shared subspace dimensionality
Goal: Determine the minimum rank needed to faithfully represent the shared structure between two models' KV caches.
Details: For layer pairs with high CKA, perform SVD on the concatenated KV tensors and measure how many singular values are needed to retain 95%/99% of variance. This gives a lower bound on the shared representation dimension k, which directly determines the compression ratio for a shared KV cache R ∈ [seq_len, k].

Idea 5: Analyze head-level similarity
Goal: Understand whether similarity is uniform across attention heads or concentrated in specific heads.
Details: Instead of flattening to [seq_len, num_head * head_dim], compute CKA per-head: for each head h, compare tensors of shape [seq_len, head_dim] across models. This reveals whether certain heads are universally shared across model sizes while others are model-specific, enabling selective head-level KV sharing.


**Script:** `cka_analysis.py`
**Output:** CKA heatmaps for each model pair, layer-to-layer alignment scores

---

## Phase 3: MLP Training

**Goal:** Learn a mapping from a smaller model's KV tensors to a larger model's KV tensors (e.g., 0.6b→1.7b, 1.7b→4b, 4b→8b).

1. Define train/val/test split over HumanEval problems
2. Architecture: MLP per layer, input = source model K or V vector, output = target model K or V vector
3. Loss: MSE between predicted and true target tensors
4. Train and track loss curves for each source→target pair
5. Evaluate on held-out test problems

**Script:** `train_mlp.py`
**Output:** Trained MLP weights per model pair, loss curves, per-layer reconstruction error

---

## Phase 4: Evaluation

**Goal:** Measure how well the transformed KV cache substitutes for the real target model's prefill cache.

1. Plug the MLP-transformed KV cache into the target model at inference time (skip real prefill, inject predicted cache)
2. Run generation and compare outputs vs. baseline (real prefill) for each source→target pair (0.6b→1.7b, 1.7b→4b, 4b→8b)
3. Metrics: exact match or downstream HumanEval pass@1

**Script:** `evaluate.py`

---

## File Structure

```
kvcache-study/
├── collect_kv.py          # Phase 1: extract KV tensors via hooks
├── cka_analysis.py        # Phase 2: CKA computation and heatmaps
├── train_mlp.py           # Phase 3: MLP training
├── evaluate.py            # Phase 4: inject transformed cache and evaluate
├── data/                  # saved KV tensors
└── results/               # plots, metrics
```
