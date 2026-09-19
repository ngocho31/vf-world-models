"""Summaries, fixed texts and the console table of the WM-12 report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .region_metrics import REGIONS
from .window_bootstrap import WindowSeries, bootstrap_indices, contrast_summary, nest, paired_summary, ratio_summary

MASKS = ("block", "random", "motion")
TARGETS = ("target_encoder", "context_encoder")
METRICS = ("cos_ln", "cos")

METRIC_NOTES = {
    "cos_ln": "cosine after layer-normalising each token over its channels (= Pearson correlation of the "
              "1024 channels); the JEPA loss compares layer-normed tokens (evaluation/losses.py:8-12)",
    "cos": "plain cosine, as evaluation/metrics.py:12-13",
    "z_star": {
        "target_encoder": "primary: ModelOutput.target_patches, the target_encoder of the Drive-JEPA checkpoint",
        "context_encoder": "secondary: context encoder tokens (Drive-JEPA 'encoder' key), the weights the EMA "
                           "pulled the training target towards",
    },
    "pooling": "token-weighted mean over all masked tokens of all windows",
    "mean_position": "baseline predicting z* by its mean at the same patch position over the evaluated windows",
    "mean_global": "baseline predicting z* by one mean vector over all tokens of the evaluated windows",
}

PREDICTOR_ROPE_NOTE = (
    "The predictor's rotary position encoding is built with grid_size = 256 // 16 = 16 and receives no grid "
    "width (src/vjepa2/src/models/predictor.py:106,240; perception/predictor.py:60-74), so each 16x32 transition "
    "is read as two 16x16 frames: rows 0-7 -> frame 2t, rows 8-15 -> frame 2t+1, and the right half of each row "
    "folds onto the next row (token 31 = row 0, col 31 is seen at h 1, w 15). Patches adjacent across column 15|16 "
    "or row 7|8 are not adjacent for the predictor; this can affect block versus scattered masks differently.")


def build_limitations(config, data: dict, epoch, top_fraction: float) -> list[str]:
    """Limits of this run, stated with the run's own numbers."""
    spacing = data["frame_spacing_s"]["median"]
    transitions = config.history_length - 1
    return [
        f"Frames are ~{spacing:.2f} s apart while training used {data['train_hz']:g} Hz data: a "
        f"{config.history_length}-frame window spans ~{transitions * spacing:.0f} s instead of "
        f"~{transitions / data['train_hz']:.1f} s, and flow between frames that far apart is much larger than in training.",
        "Whether this recording is part of the training data is not confirmed: read the scores as a comparison "
        "between masking strategies, not as held-out generalisation.",
        "Masking happens after the encoders, as in training: context tokens come from full frames, fusion attends "
        "over all motion tokens of the clip, and the confounder U averages every token of the window, masked ones "
        "included (pipeline/phase1.py:159-160, perception/predictor.py:117-122), so the context has already seen "
        "the masked region.",
        "z* is computed with the Drive-JEPA weights: the EMA-updated target encoder of training is not saved in "
        "the checkpoint.",
        f"Dynamic = raw |flow| ranking without ego-motion compensation: road close to the car and image borders "
        f"dominate it, and every transition gets its top {top_fraction:.0%} labelled dynamic even when the car is "
        "stopped (see flow_diagnostics).",
        f"The checkpoint stores epoch {epoch}.",
        "The block reference is drawn one window at a time (no batch truncation), unlike training with batch 4.",
        f"{len(data['episodes'])} recording(s) from camera {data.get('camera', 'unknown')}; consecutive windows share "
        f"{max(transitions - config.stride, 0)} of {transitions} transitions at stride {config.stride}.",
    ]


def describe_flow(flow_shape: list[int] | None, input_width: int, grid: tuple[int, int] = (16, 32)) -> str:
    if not flow_shape:
        return "not measured"
    height, width = flow_shape[-2:]
    return (f"|flow| of the {height}x{width} flow pipeline.flow_estimator returns, averaged over the "
            f"{height // grid[0]}x{width // grid[1]} flow cells covering each 16x16 input patch and multiplied by "
            f"{input_width / width:g} to give pixels of the {input_width}-pixel-wide model input")


def summarize(series: WindowSeries, n_boot: int, block: int, seed: int) -> dict:
    """Pooled scores with CIs, paired differences, and the diagnostic fractions."""
    n_windows = len(series.rows)
    resamples = {
        "ci95": bootstrap_indices(n_windows, block, n_boot, seed),
        "ci95_iid": bootstrap_indices(n_windows, 1, n_boot, seed + 1),
    }
    tree = nest({key: ratio_summary(series.pairs(key), resamples) for key in series.keys()})
    paired = {}
    for target in TARGETS:
        for metric in METRICS:
            for region in REGIONS:
                tail = f"{target}/{metric}/{region}"
                paired[f"random_minus_motion/{tail}"] = paired_summary(
                    series.pairs(f"model/random/{tail}"), series.pairs(f"model/motion/{tail}"), resamples)
                for mask in MASKS:
                    paired[f"model_minus_mean_position/{mask}/{tail}"] = paired_summary(
                        series.pairs(f"model/{mask}/{tail}"), series.pairs(f"mean_position/{mask}/{tail}"), resamples)
            for mask in MASKS:  # is the gain over the Mean-Predictor smaller on dynamic tokens than on static ones?
                pair = {f"{p}_{r}": series.pairs(f"{p}/{mask}/{target}/{metric}/{r}")
                        for p in ("model", "mean_position") for r in ("dynamic", "static")}
                paired[f"gain_dynamic_minus_static/{mask}/{target}/{metric}"] = contrast_summary(
                    [(1.0, pair["model_dynamic"]), (-1.0, pair["mean_position_dynamic"]),
                     (-1.0, pair["model_static"]), (1.0, pair["mean_position_static"])], resamples)
    return {
        "results": tree["model"],
        "baselines": {"mean_position": tree["mean_position"], "mean_global": tree["mean_global"]},
        "paired_diff": nest(paired),
        "masked_in_dynamic_fraction": tree["masked_dynamic"],
        "flow_magnitude_px": tree["flow_px"],
        "bootstrap": {"unit": "window", "n_boot": n_boot, "block_windows": block, "seed": seed,
                      "ci95": "circular block bootstrap over consecutive windows",
                      "ci95_iid": "windows resampled independently (reference only)"},
    }


def describe_data(dataset, window_ids: list[int], args) -> dict:
    """Frame count and spacing of the evaluated recording(s), read from the loader's episodes."""
    episodes = sorted({window.episode_path for window in dataset._windows})
    stamps = [np.asarray(dataset._load_episode(path)["timestamps_us"], dtype=np.int64) for path in episodes]
    gaps = np.concatenate([np.diff(s) for s in stamps]) / 1e6
    info_path = Path(args.data_root) / "dataset_info.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    return {
        "root": str(Path(args.data_root).resolve()), "split": args.split,
        "camera": info.get("camera"), "converter_target_hz": info.get("target_hz"),
        "episodes": [path.name for path in episodes], "n_frames": int(sum(s.size for s in stamps)),
        "frame_spacing_s": {"median": float(np.median(gaps)), "min": float(gaps.min()), "max": float(gaps.max())},
        "train_hz": args.train_hz, "history_length": args.history_length, "stride": args.stride,
        "n_windows_available": len(dataset), "n_windows": len(window_ids),
    }


def build_report(header: dict, summary: dict, args, config, dataset, window_ids: list[int],
                 extras: dict, masking: dict, identity: dict, flow_diagnostics: dict) -> dict:
    """The WM-12 JSON: provenance, gates, configuration, data, rules, scores, limitations."""
    gates = header["load_gates"]
    data = describe_data(dataset, window_ids, args)
    flow_shape = extras.get("flow_shape")
    motion_tokens = extras.get("motion_tokens_per_transition")
    static_tokens = extras.get("tokens_per_window", 0) // (config.history_length - 1)
    notes = [PREDICTOR_ROPE_NOTE]
    if motion_tokens is not None and motion_tokens != static_tokens:
        notes.insert(0, f"FlowFormer++ in eval mode returns (full-resolution flow, 1/8-resolution flow) and "
                        f"perception/motion_encoder.py:136-142 keeps the last element, so FlowTokenEncoder gets "
                        f"{flow_shape[-2]}x{flow_shape[-1]} flow and yields {motion_tokens} motion tokens per transition "
                        f"for fusion, against {static_tokens} static tokens (measured).")
    return {
        **header,
        "checkpoint": {**identity["checkpoint"], "drive_id": args.checkpoint_drive_id, "epoch": gates["camo"]["epoch"]},
        "flowformer": {**identity["flowformer"], "keys_loaded": gates["flowformer"]["matched"],
                       "keys_expected": gates["flowformer"]["expected"]},
        "vitl": identity["vitl"],
        "config": {"history_length": config.history_length, "stride": config.stride,
                   "image_size_wh": list(config.image_size), "seed": args.seed,
                   "ablation": {"motion_branch": config.ablation_motion_branch, "confounder": config.ablation_confounder,
                                "factorizer": config.ablation_factorizer}, **masking},
        "data": data,
        "region_rule": {"source": "pipeline.flow_estimator (FlowFormer++, frame t -> t+1), the same call the model makes",
                        "flow_shape_measured": flow_shape,
                        "magnitude": describe_flow(flow_shape, config.image_size[0]),
                        "dynamic": f"top {args.top_fraction:.0%} patches of every transition by magnitude, ties to the lower index",
                        "static": "all other patches", "ego_motion_compensated": False},
        "flow_diagnostics": flow_diagnostics,
        "observations": {"motion_tokens_per_transition": motion_tokens, "static_tokens_per_transition": static_tokens,
                         "notes": notes},
        "metric": METRIC_NOTES,
        **summary,
        "limitations": build_limitations(config, data, gates["camo"]["epoch"], args.top_fraction),
    }


def _fmt(entry: dict | None) -> str:
    if not entry or entry.get("mean") is None:
        return "n/a"
    ci = entry.get("ci95")
    return f"{entry['mean']:.4f}" + (f" [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "")


def headline_lines(summary: dict, target: str = "target_encoder", metric: str = "cos_ln") -> list[str]:
    """Console table: one line per mask x region, model vs the two mean baselines."""
    results, baselines = summary["results"], summary["baselines"]
    lines = [f"z* = {target}, metric = {metric} (mean [95% block-bootstrap CI], tokens)"]
    for mask in MASKS:
        for region in REGIONS:
            model = results[mask][target][metric][region]
            position = baselines["mean_position"][mask][target][metric][region]
            glob = baselines["mean_global"][mask][target][metric][region]
            lines.append(f"  {mask:6s} {region:7s} model {_fmt(model):32s} mean_pos {_fmt(position):32s} "
                         f"mean_glob {_fmt(glob):32s} n={model['n_tokens']}")
    for mask in MASKS:
        lines.append(f"  masked tokens in dynamic region, {mask:6s}: {_fmt(summary['masked_in_dynamic_fraction'][mask])}")
    for region in REGIONS:
        diff = summary["paired_diff"]["random_minus_motion"][target][metric][region]
        lines.append(f"  random - motion, {region:7s}: {_fmt(diff)} over {diff['n_windows']} windows")
    for mask in MASKS:
        gap = summary["paired_diff"]["gain_dynamic_minus_static"][mask][target][metric]
        lines.append(f"  gain over mean_pos, dynamic - static, {mask:6s}: {_fmt(gap)}")
    return lines
