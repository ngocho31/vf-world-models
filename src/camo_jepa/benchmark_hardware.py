"""
Hardware Benchmark for CaMo-JEPA.
Measures parameters, FLOPs, VRAM, and Latency for A100 HPC environments.
"""

import os
import json
import time
from pathlib import Path

# Redirect caches to /tmp to avoid Read-Only file system errors on HPC
os.environ.setdefault("HF_HOME", "/tmp/cache")
os.environ.setdefault("TORCH_HOME", "/tmp/cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import sys
import subprocess

# Auto-install fvcore if missing
try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    print("Warning: 'fvcore' module not found. Installing dynamically...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fvcore"])
        from fvcore.nn import FlopCountAnalysis
        HAS_FVCORE = True
        print("Successfully installed and imported fvcore.")
    except Exception as e:
        HAS_FVCORE = False
        print(f"Failed to install fvcore: {e}")

# Fix V-JEPA2 path issue for Baseline
sys.path.append(os.path.join(os.getcwd(), 'src', 'vjepa2'))

from .config import CaMoJEPAConfig
from .pipeline import CaMoJEPAPipeline
from .contracts import FrameBatch


def measure_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    ratio = trainable / total if total > 0 else 0
    return {
        "Total_M": total / 1e6,
        "Trainable_M": trainable / 1e6,
        "Frozen_M": frozen / 1e6,
        "Trainable_Ratio": ratio * 100.0,
        "Trainable_Count": trainable, # raw count
    }

def measure_flops(model, dummy_images):
    if not HAS_FVCORE:
        return {
            "Forward_GFLOPs": 0,
            "Backward_GFLOPs": 0,
            "Total_GFLOPs": 0,
        }
        
    # fvcore expects a tuple of inputs
    flops = FlopCountAnalysis(model, dummy_images)
    flops.unsupported_ops_warnings(False)
    try:
        forward_flops = flops.total()
    except Exception as e:
        print(f"fvcore FLOPs counting warning: {e}")
        forward_flops = 0
    
    # We estimate backward FLOPs as 2x Forward FLOPs for the trainable parts.
    param_info = measure_parameters(model)
    backward_flops = 2 * forward_flops * (param_info["Trainable_Ratio"] / 100.0)
    total_flops = forward_flops + backward_flops
    return {
        "Forward_GFLOPs": forward_flops / 1e9,
        "Backward_GFLOPs": backward_flops / 1e9,
        "Total_GFLOPs": total_flops / 1e9,
    }

def profile_step(model, dummy_batch, optimizer, scaler, device):
    # Warmup
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            output = model(dummy_batch)
            loss = output.losses['total'] if isinstance(output.losses, dict) else output.z_pred.sum()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    start_event = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)
    step_end = torch.cuda.Event(enable_timing=True)
    
    optimizer.zero_grad()
    
    start_event.record()
    
    # Forward
    with torch.amp.autocast('cuda', enabled=(scaler is not None)):
        output = model(dummy_batch)
        loss = output.losses['total'] if isinstance(output.losses, dict) else output.z_pred.sum()
    fwd_end.record()
    
    # Backward
    if scaler:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    bwd_end.record()
    
    # Optimizer
    if scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    step_end.record()
    
    torch.cuda.synchronize()
    
    fwd_time = start_event.elapsed_time(fwd_end)
    bwd_time = fwd_end.elapsed_time(bwd_end)
    opt_time = bwd_end.elapsed_time(step_end)
    total_time = start_event.elapsed_time(step_end)
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    reserved_vram = torch.cuda.memory_reserved() / (1024**3)
    
    return {
        "Forward_ms": fwd_time,
        "Backward_ms": bwd_time,
        "Optimizer_ms": opt_time,
        "Total_ms": total_time,
        "Peak_VRAM_GB": peak_vram,
        "Reserved_VRAM_GB": reserved_vram,
    }

def export_report(baseline_metrics, proposed_metrics, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON
    with open(out_dir / "hardware_benchmark.json", "w") as f:
        json.dump({
            "Baseline_FullFinetune": baseline_metrics,
            "Proposed_FrozenBackbone": proposed_metrics
        }, f, indent=4)
        
    # Markdown
    md = f"""# Hardware Benchmark Report: Frozen Backbone vs Full Fine-tune

| Metric Category | Metric | Baseline (Full) | Proposed (Frozen) | Improvement |
| :--- | :--- | :--- | :--- | :--- |
"""
    categories = {
        "I. Parameters": [
            ("Total_M", "Total Params (M)", "{:.1f}"),
            ("Trainable_M", "Trainable Params (M)", "{:.1f}"),
            ("Frozen_M", "Frozen Params (M)", "{:.1f}"),
            ("Trainable_Ratio", "Trainable Ratio (%)", "{:.1f}%"),
        ],
        "II. Computation": [
            ("Forward_GFLOPs", "Forward FLOPs (G)", "{:.1f}"),
            ("Backward_GFLOPs", "Estimated Backward FLOPs (G)", "{:.1f}"),
            ("Total_GFLOPs", "Total FLOPs/Step (G)", "{:.1f}"),
        ],
        "III. VRAM": [
            ("Static_VRAM_GB", "Static Model VRAM (GB)", "{:.2f}"),
            ("Peak_VRAM_GB", "Peak Training VRAM (GB)", "{:.2f}"),
            ("Activation_VRAM_GB", "Activation Memory (GB)", "{:.2f}"),
            ("Reserved_VRAM_GB", "Reserved VRAM (GB)", "{:.2f}"),
        ],
        "IV. Latency": [
            ("Forward_ms", "Forward Time (ms)", "{:.1f}"),
            ("Backward_ms", "Backward Time (ms)", "{:.1f}"),
            ("Optimizer_ms", "Optimizer Time (ms)", "{:.1f}"),
            ("Total_ms", "Total Step Latency (ms)", "{:.1f}"),
            ("Throughput", "Throughput (Samples/s)", "{:.2f}"),
            ("FPS", "Frame Throughput (FPS)", "{:.1f}"),
        ],
        "V. Network & Opt": [
            ("DDP_Sync_MB", "DDP Sync Volume (MB)", "{:.2f}"),
            ("Opt_State_MB", "Optimizer State Mem (MB)", "{:.2f}"),
        ]
    }
    
    for cat_name, metrics in categories.items():
        for i, (key, display_name, fmt) in enumerate(metrics):
            base_val = baseline_metrics.get(key, 0)
            prop_val = proposed_metrics.get(key, 0)
            base_str = fmt.format(base_val) if base_val != 0 else "N/A"
            prop_str = fmt.format(prop_val) if prop_val != 0 else "N/A"
            
            # calculate diff
            diff = ""
            if base_val != 0 and type(base_val) in (int, float) and prop_val != 0:
                ratio = (prop_val - base_val) / base_val * 100
                diff_sign = "+" if ratio > 0 else ""
                diff = f"{diff_sign}{ratio:.1f}%"
                
            cat_display = cat_name if i == 0 else ""
            md += f"| {cat_display} | {display_name} | {base_str} | {prop_str} | {diff} |\n"
            
    with open(out_dir / "hardware_benchmark_report.md", "w") as f:
        f.write(md)

def run_benchmark_scenario(config, is_frozen=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Overwrite config freezing rules safely since it's a frozen dataclass
    from dataclasses import replace
    
    # Giảm batch_size xuống 1 vì batch=2 vẫn vượt quá 40GB VRAM khi unfreeze FlowFormer++
    config = replace(config, batch_size=1)
    
    # To hit ~15-20% trainable params for "Frozen Backbone":
    # Freeze ViT (context & target encoders) but keep FlowFormer++ unfreezed
    freeze_flow = False
    
    config = replace(
        config,
        freeze_context_encoder=is_frozen,
        freeze_target_encoder=is_frozen,
        freeze_flow_estimator=freeze_flow
    )
    
    print(f"  [>] Building model (frozen={is_frozen})...")
    model = CaMoJEPAPipeline(config).to(device)
    
    static_vram = torch.cuda.memory_allocated() / (1024**3) if device.type == 'cuda' else 0
    
    param_info = measure_parameters(model)
    
    print("  [>] Generating dummy data...")
    dummy_images = torch.randn(
        config.batch_size, 
        config.history_length, 
        3, 
        config.image_size[0], 
        config.image_size[1], 
        device=device
    )
    dummy_batch = FrameBatch(images=dummy_images)
    
    print("  [>] Measuring FLOPs (fvcore)...")
    flops_info = measure_flops(model, dummy_images)
    
    print("  [>] Profiling GPU Execution (Warmup -> Forward -> Backward -> Optimizer)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    if device.type == 'cuda':
        profile_info = profile_step(model, dummy_batch, optimizer, scaler, device)
    else:
        print("  [!] CUDA not available. Skipping latency/vram profiling.")
        profile_info = {
            "Forward_ms": 0, "Backward_ms": 0, "Optimizer_ms": 0, "Total_ms": 0,
            "Peak_VRAM_GB": 0, "Reserved_VRAM_GB": 0
        }
    
    # Clean up model
    del model
    del optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    metrics = {}
    metrics.update(param_info)
    metrics.update(flops_info)
    metrics.update(profile_info)
    
    metrics["Static_VRAM_GB"] = static_vram
    metrics["Activation_VRAM_GB"] = metrics["Peak_VRAM_GB"] - static_vram if metrics["Peak_VRAM_GB"] > 0 else 0
    
    batch_size = config.batch_size
    history_length = config.history_length
    total_sec = metrics["Total_ms"] / 1000.0
    metrics["Throughput"] = batch_size / total_sec if total_sec > 0 else 0
    metrics["FPS"] = (batch_size * history_length) / total_sec if total_sec > 0 else 0
    
    trainable_count = param_info["Trainable_Count"]
    metrics["DDP_Sync_MB"] = trainable_count * 4 / (1024**2)
    metrics["Opt_State_MB"] = trainable_count * 8 / (1024**2)
    
    return metrics


def main():
    config = CaMoJEPAConfig()
    
    # By default, output_log_dir might be relative. Let's make sure it's valid.
    out_dir = Path(config.output_log_dir).expanduser().resolve()
    print(f"==================================================")
    print(f" CaMo-JEPA Hardware Benchmark")
    print(f" Output Directory: {out_dir}")
    print(f"==================================================")
    
    print("\n[1/2] Running Baseline (Full Fine-tune)...")
    try:
        baseline_metrics = run_benchmark_scenario(config, is_frozen=False)
    except Exception as e:
        print(f"  [Error] Baseline run failed. Using empty metrics. Exception: {e}")
        baseline_metrics = {}
        
    print("\n[2/2] Running Proposed (Frozen Backbone)...")
    try:
        proposed_metrics = run_benchmark_scenario(config, is_frozen=True)
    except Exception as e:
        print(f"  [Error] Proposed run failed. Using empty metrics. Exception: {e}")
        proposed_metrics = {}
    
    print("\n[>] Exporting Benchmark Report...")
    export_report(baseline_metrics, proposed_metrics, out_dir)
    print(f"[Done] Report successfully saved to:")
    print(f"       - {out_dir}/hardware_benchmark_report.md")
    print(f"       - {out_dir}/hardware_benchmark.json")


if __name__ == "__main__":
    main()
