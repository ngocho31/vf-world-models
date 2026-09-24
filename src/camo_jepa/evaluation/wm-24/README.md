## Formula
CKA(X,Y) = ||X^T Y||²_F / (||X^T X||_F · ||Y^T Y||_F)

## Running the Evaluation

```bash
python wm-24/eval_layerwise_cka.py \
    --checkpoint path/to/camo_jepa.pt \
    --dataset-root path/to/dataset \
    --output results/layerwise_cka.json
```

## Expected Outputs
- `results/layerwise_cka.json`: Contains the full CKA matrix, diagonal values, and layer divergence summaries.
- `results/layerwise_cka_chart.png`: 2D heatmap showing the correspondence across all layers between CaMo-JEPA and V-JEPA2.