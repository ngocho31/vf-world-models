# vf-world-models
Implementation of VF World Models based on Drive-JEPA

## Qualitative Validation with VF data

### Download checkpoints

```bash
cd /workspace

mkdir -p .cache/checkpoints/vjepa2
wget -4 -c -N -O .cache/checkpoints/vjepa2/vitl_merge_3dataset_e50.pt \
  https://huggingface.co/datasets/LinhanWang/Drive-JEPA/resolve/main/vitl_merge_3dataset_e50.pt

mkdir -p .cache/checkpoints/phase2
wget -4 -c -N -O .cache/checkpoints/phase2/drive_jepa_perception_free_agent_vitl.ckpt \
  https://huggingface.co/datasets/LinhanWang/Drive-JEPA/resolve/main/drive_jepa_perception_free_agent_vitl.ckpt
```

### Run

```bash
cd /workspace
source ./src/env.local.sh
./src/navsim/scripts/evaluation/run_vf_qualitative_eval.sh
```

### Output

```
outputs/vf_qualitative_eval/
├── vf_qualitative_eval.mp4
├── frames/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
└── predicted_trajectories.json      # list of records:
                                     #   frame, prev_frame, trajectory_xyh [[x,y,θ]×8]
```
