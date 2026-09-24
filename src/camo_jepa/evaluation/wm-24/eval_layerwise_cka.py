import sys
from pathlib import Path

import random, subprocess, datetime, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import math

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None

def check_nan_inf(t: torch.Tensor, name: str = "") -> list[str]:
    w = []
    if torch.isnan(t).any():
        w.append(f"{name}: {int(torch.isnan(t).sum())} NaN")
    if torch.isinf(t).any():
        w.append(f"{name}: {int(torch.isinf(t).sum())} Inf")
    return w

def move_batch_to_device(batch, device):
    batch.images = batch.images.to(device)
    if batch.ego_motion is not None:
        batch.ego_motion = batch.ego_motion.to(device)
    return batch

def save_results(path: str, data: dict, args) -> None:
    from pathlib import Path
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": data.get("experiment", "unknown"),
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "checkpoint": getattr(args, "checkpoint", None),
        "dataset": getattr(args, "dataset_root", None),
        "device": getattr(args, "device", None),
        "batch_size": getattr(args, "batch_size", None),
        "seed": getattr(args, "seed", None),
        "config": {k: v for k, v in vars(args).items() if _serializable(v)},
        "metrics": data.get("metrics", {}),
        "warnings": data.get("warnings", []),
    }
    for k in data:
        if k not in payload:
            payload[k] = data[k]
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, default=str)

def _serializable(v):
    try: json.dumps(v); return True
    except: return False

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="CaMo-JEPA checkpoint")
    p.add_argument("--dataset-root", default="dataset_camo/vinfast")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", required=True, help="JSON output path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--vitl-checkpoint", default=None)
    p.add_argument("--flowformer-checkpoint", default=None)
    
    p.add_argument("--vjepa2-checkpoint", default=".cache/checkpoints/vjepa2/vitl_merge_3dataset_e50.pt", type=str)
    p.add_argument("--vjepa2-checkpoint-key", default="target_encoder", type=str)
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--heatmap-output", type=str, default=None)
    return p

def load_camo_model(args):
    from src.camo_jepa.config import CaMoJEPAConfig
    from src.camo_jepa.pipeline.phase1 import CaMoJEPAPipeline
    from src.camo_jepa.pipeline.checkpoints import load_checkpoint
    from src.camo_jepa.data.camo import make_camo_dataloader
    
    kwargs = {"dataset_root": args.dataset_root, "dataset_split": args.dataset_split,
              "batch_size": args.batch_size, "shuffle": False}
    if args.vitl_checkpoint:
        kwargs["vitl_checkpoint_path"] = args.vitl_checkpoint
    if args.flowformer_checkpoint:
        kwargs["motion_estimator_checkpoint_path"] = args.flowformer_checkpoint
    config = CaMoJEPAConfig(**kwargs)
    model = CaMoJEPAPipeline(config)
    load_checkpoint(args.checkpoint, model, strict=False)
    model.to(args.device).eval()
    loader = make_camo_dataloader(config)
    return model, loader, config

def load_vjepa2_model(checkpoint_path, device, repo_root, checkpoint_key="target_encoder"):
    vjepa2_src = str(Path(repo_root) / "src" / "vjepa2" / "src")
    if vjepa2_src not in sys.path:
        sys.path.insert(0, vjepa2_src)
    from models.vision_transformer import vit_large
    
    model = vit_large(img_size=(256, 512), patch_size=16, num_frames=2,
                      tubelet_size=2, use_rope=True, uniform_power=True,
                      handle_nonsquare_inputs=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get(checkpoint_key, ckpt)
    cleaned = {}
    for k, v in sd.items():
        nk = k
        while nk.startswith("module.") or nk.startswith("backbone."):
            nk = nk.replace("module.", "", 1).replace("backbone.", "", 1)
        cleaned[nk] = v
    model.load_state_dict(cleaned, strict=False)
    return model.to(device).eval()

def build_vjepa2_clips(images):
    """Convert FrameBatch images [B,T,C,H,W] to V-JEPA2 clips [B*(T-1),C,2,H,W]."""
    B, T, C, H, W = images.shape
    clips = torch.stack((images[:, :-1], images[:, 1:]), dim=2)
    clips = clips.permute(0, 1, 3, 2, 4, 5)
    return clips.reshape(B * (T - 1), C, 2, H, W)

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X.float(); Y = Y.float()
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic_xy = (X.T @ Y).norm('fro') ** 2
    hsic_xx = (X.T @ X).norm('fro') ** 2
    hsic_yy = (Y.T @ Y).norm('fro') ** 2
    denom = hsic_xx.sqrt() * hsic_yy.sqrt() + 1e-10
    return (hsic_xy / denom).item()

def extract_layerwise_features(backbone, dataloader, device, layer_indices, max_samples):
    original_out_layers = getattr(backbone, 'out_layers', None)
    backbone.out_layers = layer_indices
    
    features = {idx: [] for idx in layer_indices}
    samples_collected = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_collected >= max_samples:
                break
                
            batch = move_batch_to_device(batch, device)
            images = batch.images
            
            clips = build_vjepa2_clips(images)
            n_clips = clips.shape[0]
            
            out = backbone(clips)
            
            if isinstance(out, (list, tuple)):
                for i, idx in enumerate(layer_indices):
                    pooled = out[i].mean(dim=1).cpu() 
                    features[idx].append(pooled)
            elif isinstance(out, dict):
                for idx in layer_indices:
                    pooled = out[idx].mean(dim=1).cpu()
                    features[idx].append(pooled)
            else:
                pooled = out.mean(dim=1).cpu()
                features[layer_indices[-1]].append(pooled)
                
            samples_collected += n_clips
            
    if original_out_layers is not None:
        backbone.out_layers = original_out_layers
        
    final_features = {}
    for idx in layer_indices:
        if len(features[idx]) > 0:
            final_features[idx] = torch.cat(features[idx], dim=0)[:max_samples]
        else:
            final_features[idx] = torch.empty(0)
    
    return final_features

def generate_heatmap(cka_matrix, layer_names_x, layer_names_y, heatmap_path, diagonal_cka, divergence_summary):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not installed. Skipping heatmap generation.")
        return
        
    plt.figure(figsize=(10, 8))
    plt.imshow(cka_matrix, cmap='magma', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(label='CKA')
    
    plt.title('Layer-wise CKA: CaMo-JEPA vs V-JEPA2', pad=20)
    plt.xlabel('V-JEPA2 Layer')
    plt.ylabel('CaMo-JEPA Layer')
    
    plt.xticks(ticks=range(len(layer_names_x)), labels=layer_names_x, rotation=90, fontsize=8)
    plt.yticks(ticks=range(len(layer_names_y)), labels=layer_names_y, fontsize=8)
    
    textstr = '\n'.join([
        f"Overall Mean: {divergence_summary['overall_mean']:.4f}",
        f"Early Mean (0-7): {divergence_summary['early_mean']:.4f}",
        f"Mid Mean (8-15): {divergence_summary['mid_mean']:.4f}",
        f"Late Mean (16-23): {divergence_summary['late_mean']:.4f}"
    ])
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.gca().text(1.05, 0.5, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='center', bbox=props)
            
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    
    print("Loading CaMo-JEPA model...")
    camo_model, dataloader, config = load_camo_model(args)
    camo_backbone = camo_model.context_encoder.backbone
    
    num_layers = len(camo_backbone.blocks) if hasattr(camo_backbone, 'blocks') else 24
    layer_indices = list(range(num_layers))
    
    print(f"Loading V-JEPA2 model from {args.vjepa2_checkpoint}...")
    vjepa2_model = load_vjepa2_model(args.vjepa2_checkpoint, args.device, _REPO_ROOT, args.vjepa2_checkpoint_key)
    
    print(f"Extracting features from both models (max samples: {args.max_samples})...")
    camo_features = extract_layerwise_features(camo_backbone, dataloader, args.device, layer_indices, args.max_samples)
    vjepa2_features = extract_layerwise_features(vjepa2_model, dataloader, args.device, layer_indices, args.max_samples)
    
    print("Computing CKA matrix...")
    cka_matrix = np.zeros((num_layers, num_layers))
    
    for i in range(num_layers):
        for j in range(num_layers):
            if i in camo_features and j in vjepa2_features and len(camo_features[i]) > 0 and len(vjepa2_features[j]) > 0:
                cka_matrix[i, j] = linear_cka(camo_features[i], vjepa2_features[j])
                
    diagonal_cka = [cka_matrix[i, i] for i in range(num_layers)]
    
    early_mean = np.mean(diagonal_cka[0:8]) if num_layers >= 8 else 0
    mid_mean = np.mean(diagonal_cka[8:16]) if num_layers >= 16 else 0
    late_mean = np.mean(diagonal_cka[16:24]) if num_layers >= 24 else 0
    overall_mean = np.mean(diagonal_cka)
    
    divergence_summary = {
        "early_mean": float(early_mean),
        "mid_mean": float(mid_mean),
        "late_mean": float(late_mean),
        "overall_mean": float(overall_mean),
        "min_val": float(np.min(diagonal_cka)),
        "min_idx": int(np.argmin(diagonal_cka)),
        "max_val": float(np.max(diagonal_cka)),
        "max_idx": int(np.argmax(diagonal_cka))
    }
    
    print(f"\nLayer-wise CKA Analysis ({num_layers} layers)")
    print("════════════════════════════════════")
    print("Diagonal CKA (layer-to-layer correspondence):")
    print(f"  Early (0-7):  {early_mean:.4f}")
    print(f"  Middle (8-15): {mid_mean:.4f}")
    print(f"  Late (16-23):  {late_mean:.4f}")
    print(f"\nOverall diagonal mean: {overall_mean:.4f}")
    print(f"Min diagonal CKA: Layer {divergence_summary['min_idx']} = {divergence_summary['min_val']:.4f}")
    print(f"Max diagonal CKA: Layer {divergence_summary['max_idx']} = {divergence_summary['max_val']:.4f}")
    
    heatmap_path = args.heatmap_output
    if not heatmap_path:
        heatmap_path = str(Path(args.output).with_suffix("")) + "_chart.png"
        
    layer_names = [f"L{i}" for i in range(num_layers)]
    Path(heatmap_path).parent.mkdir(parents=True, exist_ok=True)
    generate_heatmap(cka_matrix, layer_names, layer_names, heatmap_path, diagonal_cka, divergence_summary)
    
    results = {
        "experiment": "Layer-wise CKA",
        "classification": "Comparable with V-JEPA2",
        "metrics": {
            "cka_matrix": cka_matrix.tolist(),
            "diagonal_cka": diagonal_cka,
            "divergence_summary": divergence_summary,
            "layer_names": layer_names
        }
    }
    save_results(args.output, results, args)
    print(f"\nSaved results to {args.output} and heatmap to {heatmap_path}")

if __name__ == "__main__":
    main()