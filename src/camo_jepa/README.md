# CaMo-JEPA

CaMo-JEPA is a Phase 1 video representation pipeline for synchronized driving
images and ego motion. It combines a frozen Drive-JEPA V-JEPA2 ViT-L backbone
with frame-difference motion features, latent factorization, confounder estimation,
and future-latent prediction.

## Module Hierarchy

```text
camo_jepa/
  causal/
    latent.py               # Contain LatentFactorizer (create z_task, z_exogenous)
    confounder.py           # Contain ConfounderGRU (create vector U_t)
  data/
    camo.py                 # Dataloader, transformations, return FrameBatch
  evaluation/
    losses.py               # L_JEPA, L_orth, L_recon
    metrics.py              # Evaluate intrinsic metrics (Cosine similarity, Mutual Information bounds)
  perception/
    static_encoder.py       # Load & freeze Drive-JEPA ViT-L (Context & Target)
    motion_encoder.py       # Load pre-trained frozen motion checkpoint
    fusion.py               # GatedCrossAttentionFusion combine static & dynamic
    masking.py              # MaskSampler (create mask for z_task)
    predictor.py            # Load checkpoint V-JEPA predictor & Causal Wrapper
  pipeline/
    phase1.py               # CaMoJEPAPipeline (organize all modules into a single forward pass)
    engine.py               # train_step() (Forward -> Loss -> Backward -> Optimizer step -> EMA update)
    checkpoints.py          # Save/Load weights
  cli.py                    # Entry point to run the script
  config.py                 # CaMoJEPAConfig hyperparameter definitions
  contracts.py              # Schema, Data classes (FrameBatch, Output structure)
```

### Root Level

* `config.py`: Defines `CaMoJEPAConfig` containing all hyperparameters, paths, loss weights ($\lambda_1, \lambda_2$), and dimension settings.
* `contracts.py`: Dataclasses defining standardized interfaces across modules (FrameBatch and ModelOutput).
* `cli.py`: Entry point for end-to-end execution. Loads configuration, batch, initializes CaMoJEPAPipeline, performs a complete train_step(), and saves model checkpoints.

### `data/` (Xử lý Dữ liệu)

* `camo.py`: converts transformed `.npz` episodes and raw images into sliding window `FrameBatch` samples.

### `motion/` (Động lực Hình ảnh)

* `flow.py`: LiteFlowNet3 (Not implemented yet).

### `perception/` (Nhận thức Cốt lõi)

* `static_encoder.py`: Loads frozen Drive-JEPA ViT-L encoders (`context_encoder` and `target_encoder`) directly from V-JEPA2 source and checkpoint loader, and returns patch tokens `[B, T-1, N, 1024]`.

* `motion_encoder.py`: Loads frozen Motion ViT-L encoder from Motion Checkpoint, returns patch tokens `[B, T-1, N, 1024]`.

* `fusion.py`: Gated residual fusion ($z_t = e_{\text{static}} + \gamma \cdot \tanh(g) \odot e_{\text{dynamic}}$).

* `masking.py`: Wraps V-JEPA2 `_MaskGenerator` from `vf-drive-jepa/vjepa2/src/masks/multiseq_multiblock3d.py` and generates spatiotemporal block masks directly on flattened $z_{\text{task}}$ tokens.

* `predictor.py`: Wraps the base V-JEPA2 predictor from source, loads predictor weights with the V-JEPA2 checkpoint loader, and injects confounder vector $U$.

### `causal/` (Cơ chế Nhân quả)

* `latent.py`: Disentangles fused features into $z_{\text{task}}$ and $z_{\text{exogenous}}$.

* `confounder.py`: N-step GRU tracking history into confounder vector $U$.

#### `evaluation/` (Hàm Tổn thất & Đánh giá)

* `losses.py`: computes composite loss $\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{JEPA}} + \lambda_2 \mathcal{L}_{\text{orth}} + \lambda_3 \mathcal{L}_{\text{recon}}$.

* `metrics.py`: computes non-differentiable metrics for offline evaluation (cosine similarity, mutual information bounds).

### `pipeline/` (Lắp ráp Tổng thể)

* `phase1.py`: `CaMoJEPAPipeline` orchestrates the forward pass across all modules.

* `engine.py`: `train_step()` performs forward pass, calculates loss, executes backpropagation (`loss.backward()`), updates optimizer weights, and calls EMA updates on the target encoder.

* `checkpoints.py`: Saves/loads CaMo-JEPA trainable weights (Flow Encoder, Fusion, Factorizer, Confounder, Causal Predictor). Frozen Motion encoder and Frozen ViT encoders are not saved.

### `evaluation/` (Đánh giá)

* `losses.py`: computes composite loss $\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{JEPA}} + \lambda_2 \mathcal{L}_{\text{orth}} + \lambda_3 \mathcal{L}_{\text{recon}}$.
* `metrics.py`: computes non-differentiable metrics for offline evaluation (cosine similarity, mutual information bounds).

## Pipeline

```text
==================================================================================================
                                  1. ĐẦU VÀO DỮ LIỆU (INPUT BATCH)
==================================================================================================

                            FrameBatch: images [B, T, 3, H, W]
                                          │
                                          ▼ (Ghép các cặp khung hình t và t+1)
                                   Clips (2-frames)
                                 [B, T-1, 2, 3, H, W]
                  ┌───────────────────────┴───────────────────────┐
                  │                                               │
==================┼===============================================┼===============================
                  │                2. PERCEPTION BRANCHES         │
==================┼===============================================┼===============================
                  │                                               │
       ┌──────────┴──────────┐                                    │
       ▼                     ▼                                    ▼
[ Context Encoder ]   [ Target Encoder ]                 [ Flow Estimator ]
 (ViT-L, FROZEN)     (ViT-L, FROZEN+EMA)                (FlowFormer++, FROZEN)
       │                     │                                    │
       │                     │                                    ▼
       │                     │                         [ FlowTokenEncoder ]
       │                     │                          (Conv2d stride=16, TRAINABLE)
       │                     │                                    │
       ▼                     ▼                                    ▼
 static_patches        target_patches                      dynamic_patches
[B, T-1, N, 1024]    [B, T-1, N, 1024]                    [B, T-1, N, 1024]
       │                                                          │
=======┼==========================================================┼===============================
       │                          3. FUSION & FACTORIZATION       │
       └──────────────┐             ┌─────────────────────────────┘
                      ▼             ▼
            [ GatedCrossAttentionFusion Module ]
             (Cross-Attention/Gate, TRAINABLE)
                      │
                      ▼
               fused_patches [B, T-1, N, 1024]
                      │
       ┌──────────────┴─────────────────────────┐
       ▼ (Spatial Mean-Pool over N)             ▼
  fused_global                         [ LatentFactorizer ]
 [B, T-1, 1024]                          (Linear, TRAINABLE)
       │                                        │
       ▼                                ┌───────┴─────────────────────────┐
[ ConfounderGRU ]                       ▼                                 ▼
 (GRU, TRAINABLE)               z_task_patches                       z_exogenous
       │                       [B, T-1, N, 1024]                    [B, T-1, N, 1024]
       │                                │                                 │
       │                                ▼                                 │
       │                         [ MaskSampler ]                          │
       │                     (3D Block Masking V-JEPA2)                   │
       │                                │                                 │
       │                 ┌──────────────┴──────────────┐                  │
       ▼                 ▼                             ▼                  │
   Confounder      z_task_masked                   mask_indices           │
   Vector U        [B, K, 1024]                       [B, M]              │
   [B, 128]              │                             │                  │
       │                 │                             ▼                  │
       │                 │                     Extract patches            │
       │                 │                    from target_patches         │
       │                 │                             │                  │
       │                 │                             ▼                  │
       │                 │                   z_target [B, M, 1024]        │
       │                 │                             │                  │
=======┼=================┼=============================┼==================┼=======================
       │                 │        4. PREDICTION        │                  │
       └────────┐        │                             │                  │
                ▼        ▼                             │                  │
          [ CausalPredictor ]                          │                  │
        (Transformer, TRAINABLE)                       │                  │
                │                                      │                  │
                ▼                                      │                  │
       z_pred [B, M, 1024]                             │                  │
                │                                      │                  │
================┼======================================┼==================┼=======================
                │              5. LOSS & OPTIMIZATION  │                  │
                └──────────────────────┬───────────────┘                  │
                                       ▼                                  ▼
                              ====================
                              |   CaMoJEPALoss   |
                              ====================
                              │ 1. L_JEPA  (Smooth L1: z_pred  vs  z_target)
                              │ 2. L_orth  (Cosine Orth: z_task vs z_exogenous)
                              │ 3. L_recon (Cosine Sim: fused   vs static)
                              ====================
                                       │
                                       ▼
                                   total_loss
                                       │
                                       ▼
                            [ total_loss.backward() ]
                                       │
                                       ▼
                               [ optimizer.step() ]
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
   Update Gradient            Update Gradient            Update Gradient
  - FlowTokenEncoder         - GatedCrossAttentionFusion - ConfounderGRU
  - LatentFactorizer         - CausalPredictor           - Projection layers
                                       │
                                       ▼
                         [ update_target_encoder() ]
                                       │
                                       ▼
                         Update EMA weights:
                         Target Encoder ← τ * Target Encoder + (1-τ) * Context Encoder
==================================================================================================
```

## Requirements

Default required assets, relative to the repository root:

```text
.cache/checkpoints/vjepa2/vitl_merge_3dataset_e50.pt
.cache/checkpoints/motion_estimator/kitti_finetune_vf.pth
src/vjepa2/
src/flowformer/
dataset_camo/<source_name>
```

## Converted Dataset

`dataset_root` can name a single source such as `dataset_camo/navsim`, or the
aggregate `dataset_camo` directory containing sources such as `navsim/` and
`vf/`.

```text
dataset_camo/navsim/
  dataset_info.json
  manifest.jsonl
  episodes/
    trainval/<log_name>.npz
    test/<log_name>.npz
  images/
    trainval/<log_name>/<token>.jpg
    test/<log_name>/<token>.jpg
```

Each episode contains equal-length arrays:

```text
timestamps_us          int64   [T]
source_timestamps_us   int64   [T]
tokens                 string  [T]
image_paths            string  [T]
can_bus                float32 [T, 18]
ego_motion             float32 [T, 2]
ego_motion_names       string  [2]
```

`ego_motion` is observed motion, not direct Drive-JEPA control output:

```text
[yaw_rate_radps, longitudinal_acceleration_mps2]
```

## Smoke Run

From the repository root, with the configured dataset and checkpoint paths
available:

```bash
python3 -m src.camo_jepa.cli
```
