# Hoàn Chỉnh: Đánh Giá Không Gian Ẩn (Spatial Latent Evaluation) - Toàn Bộ 7 Kỹ Thuật
Tài liệu này tổng hợp toàn bộ code từ cả hai phase: (1) Đánh giá OOD cơ bản, và (2) Các thuật toán không gian nâng cao (PCA, Occlusion, BLS). Bạn có thể copy trực tiếp vào Google Colab để chạy.

## Mục Lục
1. Tổng Quan Kiến Trúc
2. Chuẩn Bị Môi Trường (Colab)
3. Script 1: Trích Xuất Token Features
4. Script 2: Cosine Similarity + OOD Scores
5. Script 3: Mahalanobis Distance
6. Script 4: Spatial Attention Heatmaps
7. Script 5: PCA Visualization
8. Script 6: Occlusion Mapping
9. Script 7: Background Leakage Score (BLS)
10. Script Tổng Hợp: Chạy Toàn Bộ Pipeline
11. Bảng Tổng Hợp Sản Phẩm Đầu Ra (Deliverables)

## 1. Tổng Quan Kiến Trúc

```
Ảnh gốc (H×W×3)
    │
    ▼
┌─────────────────────────┐
│  Patch Embedding (P=16) │  Chia ảnh thành N = (H/16)×(W/16) mảnh
│  + Positional Embedding │  Mỗi mảnh → 1 token vector ∈ R^1024
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  24 × Transformer Block │  Self-Attention học tương quan không gian
│  (ViT-Large, D=1024)    │  giữa tất cả các token (patch ↔ patch)
└─────────────────────────┘
    │
    ▼
  Token Features z ∈ R^{N×1024}   ← ĐÂY LÀ THỨ CHÚNG TA TRÍCH XUẤT
    │
    ├──→ Cosine Similarity  (so sánh từng cặp token)
    ├──→ Mahalanobis Distance (đo OOD so với phân phối pre-trained)
    └──→ Attention Weights   (trọng số chú ý từ block cuối)
```

**File nguồn chính trong project:**
- Encoder ViT: `src/vf-drive-jepa/vjepa2/src/models/vision_transformer.py`
- Attention Block: `src/vf-drive-jepa/vjepa2/src/models/utils/modules.py`
- Model wrapper: `src/vf-drive-jepa/navsim/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py`
- Agent config: `src/vf-drive-jepa/navsim/navsim/planning/script/config/common/agent/drive_jepa_perception_free_agent.yaml`

---

## 2. Chuẩn Bị Môi Trường (Colab)

Tạo một ô code trên Colab và chạy các lệnh sau (nếu bạn đã chạy Notebook trước đó thì nhiều bước đã xong, chỉ cần đảm bảo clone repo + cài đặt đầy đủ):

```bash
# Clone repository (bỏ qua nếu đã có)
!git clone https://github.com/NguyenKhanh2603/vinfast.git
%cd /content/vinfast

# Cài thư viện
!pip install -r src/vf-drive-jepa/navsim/requirements.txt
!pip install -r src/vf-drive-jepa/vjepa2/requirements.txt
!pip install nuplan-devkit pytorch-lightning aioboto3 retry pyquaternion hydra-core

# Tải checkpoint encoder V-JEPA2
!mkdir -p .cache/checkpoints/vjepa2
!wget -4 -c -N -O .cache/checkpoints/vjepa2/vitl_merge_3dataset_e50.pt \
  https://huggingface.co/datasets/LinhanWang/Drive-JEPA/resolve/main/vitl_merge_3dataset_e50.pt

# Giải nén dữ liệu ảnh
!rm -rf dataset_vf/data/CAMERA/CAM_P_F
!mkdir -p dataset_vf/data/CAMERA/CAM_P_F
!tar -xzf data/CAM_P_F_800_1s.tar.gz -C dataset_vf/data/CAMERA/CAM_P_F
!find dataset_vf/data/CAMERA/CAM_P_F -mindepth 2 -type f -name "*.jpg" \
  -exec mv {} dataset_vf/data/CAMERA/CAM_P_F/ \;

# Thiết lập biến môi trường
!sed -i 's|export WORKSPACE_ROOT="/workspace"|export WORKSPACE_ROOT="/content/vinfast"|g' \
  src/vf-drive-jepa/env.local.sh
!sed -i 's|export VF_DRIVE_JEPA_ROOT="${WORKSPACE_ROOT}/vf-drive-jepa"|export VF_DRIVE_JEPA_ROOT="${WORKSPACE_ROOT}/src/vf-drive-jepa"|g' \
  src/vf-drive-jepa/env.local.sh
```

---

## 3. Script 1: Trích Xuất Token Features

> **Mục đích:** Nạp encoder V-JEPA2 pre-trained, đưa ảnh từ camera VinFast vào,
> trích xuất ra ma trận token features `z ∈ R^{N×1024}` cho mỗi ảnh.

Tạo file `spatial_eval/extract_features.py` hoặc paste vào 1 ô Colab:

```python
#!/usr/bin/env python3
"""
Script 1: Trích xuất Token Features từ V-JEPA2 Encoder (ViT-Large).
Output: file .pt chứa dict { "features": Tensor[num_images, N_tokens, 1024],
                              "filenames": List[str] }
"""
import sys
import os

# Thiết lập PYTHONPATH cho Colab
WORKSPACE = "/content/vinfast"
VF_DRIVE_JEPA = f"{WORKSPACE}/src/vf-drive-jepa"
sys.path.insert(0, VF_DRIVE_JEPA)
sys.path.insert(0, f"{VF_DRIVE_JEPA}/navsim")
sys.path.insert(0, f"{VF_DRIVE_JEPA}/vjepa2")
os.chdir(VF_DRIVE_JEPA)

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

# --- Import encoder giống hệt cách DriveJEPAModel load ---
import yaml
from vjepa2.evals.image_classification_frozen.modelcustom.vit_encoder import init_module


def load_vjepa2_encoder(
    checkpoint_path: str,
    resolution=(256, 512),
    config_path: str = "./vjepa2/configs/eval/vitl/in1k.yaml",
) -> torch.nn.Module:
    """
    Load encoder ViT-Large từ checkpoint V-JEPA2.
    Tham số resolution=(256, 512) khớp với cấu hình front_only=True
    trong drive_jepa_perception_free_agent.yaml.
    """
    with open(config_path, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    model_kwargs = params["model_kwargs"]
    wrapper_kwargs = model_kwargs["wrapper_kwargs"]
    # Agent config dùng double_image nên num_frames=2
    wrapper_kwargs["img_as_video_nframes"] = 2
    model_kwargs = model_kwargs["pretrain_kwargs"]
    model_kwargs["encoder"]["model_name"] = "vit_large"

    encoder = init_module(
        resolution=resolution,
        checkpoint=checkpoint_path,
        model_kwargs=model_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        # register_prehook=True: tự động nhân đôi ảnh thành video 2 frame
        register_prehook=True,
    )
    encoder.eval()
    return encoder


def preprocess_image(image_path: str, resolution=(256, 512)) -> torch.Tensor:
    """
    Tiền xử lý ảnh giống cách DriveJEPAModel.make_transform() làm:
    1. Resize về resolution
    2. Chuyển thành tensor [0, 1]
    3. Chuẩn hóa ImageNet (mean, std)
    """
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),  # [C, H, W] ∈ [0, 1]
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return transform(img)  # [3, 256, 512]


@torch.no_grad()
def extract_features(
    encoder: torch.nn.Module,
    image_dir: str,
    device: str = "cuda",
    max_images: int = 50,
) -> dict:
    """
    Trích xuất token features cho tất cả ảnh trong thư mục.

    Returns:
        dict với keys:
        - "features": Tensor [num_images, N_tokens, D=1024]
        - "filenames": List[str]
    """
    encoder = encoder.to(device)
    image_paths = sorted(Path(image_dir).glob("*.jpg"))[:max_images]

    all_features = []
    filenames = []

    for i, img_path in enumerate(image_paths):
        # Tiền xử lý: [3, 256, 512]
        img_tensor = preprocess_image(str(img_path))
        # Thêm batch dim: [1, 3, 256, 512]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Forward qua encoder ViT
        # Prehook sẽ tự chuyển [B, C, H, W] → [B, C, 2, H, W] (video 2 frames)
        # Output: [B, N_tokens, D=1024]
        # Với resolution (256, 512) và patch_size=16, tubelet_size=2:
        #   N = (2/2) × (256/16) × (512/16) = 1 × 16 × 32 = 512 tokens
        token_features = encoder(img_tensor)  # [1, 512, 1024]

        all_features.append(token_features.cpu())
        filenames.append(img_path.name)

        if (i + 1) % 10 == 0:
            print(f"[INFO] Extracted {i+1}/{len(image_paths)} images")

    features_tensor = torch.cat(all_features, dim=0)  # [num_images, 512, 1024]
    print(f"[DONE] Feature shape: {features_tensor.shape}")
    print(f"       Mỗi ảnh → {features_tensor.shape[1]} tokens × {features_tensor.shape[2]}D")

    return {"features": features_tensor, "filenames": filenames}


# ======================== CHẠY ========================
if __name__ == "__main__":
    ENCODER_CKPT = f"{WORKSPACE}/.cache/checkpoints/vjepa2/vitl_merge_3dataset_e50.pt"
    CAMERA_DIR = f"{WORKSPACE}/dataset_vf/data/CAMERA/CAM_P_F"
    OUTPUT_PATH = f"{WORKSPACE}/outputs/spatial_eval/vf_token_features.pt"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # 1. Load encoder
    encoder = load_vjepa2_encoder(ENCODER_CKPT)
    print(f"[INFO] Encoder loaded. Embed dim = {encoder.embed_dim}")

    # 2. Extract features
    result = extract_features(encoder, CAMERA_DIR, device=device, max_images=50)

    # 3. Lưu kết quả
    torch.save(result, OUTPUT_PATH)
    print(f"[SAVED] {OUTPUT_PATH}")
```

---

## 4. Script 2: Cosine Similarity + OOD Scores

> **Mục đích:** Tính ma trận Cosine Similarity giữa tất cả các token trong một ảnh
> (Intra-image), và giữa token trung bình của các ảnh khác nhau (Inter-image).
> Phát hiện token/ảnh OOD dựa trên ngưỡng CosSim thấp.

```python
#!/usr/bin/env python3
"""
Script 2: Tính Cosine Similarity và phát hiện OOD tokens.
Input: file .pt từ Script 1
Output: ma trận OOD scores + heatmap CSV
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = "/content/vinfast"
OUTPUT_DIR = f"{WORKSPACE}/outputs/spatial_eval"


def compute_cosine_similarity_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Tính ma trận Cosine Similarity giữa tất cả tokens trong 1 ảnh.

    Công thức (từ Đánh_giá_1.docx):
        CosSim(u, v) = <u, v> / (||u||₂ · ||v||₂)

    Args:
        features: [N_tokens, D] — token features của 1 ảnh
    Returns:
        sim_matrix: [N_tokens, N_tokens] — ma trận cosine similarity
    """
    # Chuẩn hóa L2 từng token
    features_norm = F.normalize(features, p=2, dim=-1)  # [N, D]
    # Nhân ma trận → cosine similarity
    sim_matrix = features_norm @ features_norm.T  # [N, N]
    return sim_matrix


def compute_inter_image_cosine(features: torch.Tensor) -> torch.Tensor:
    """
    Tính Cosine Similarity giữa đặc trưng trung bình (mean pooled)
    của từng ảnh với nhau.

    Args:
        features: [num_images, N_tokens, D]
    Returns:
        sim_matrix: [num_images, num_images]
    """
    # Mean pooling: lấy trung bình token features → 1 vector đại diện/ảnh
    img_features = features.mean(dim=1)  # [num_images, D]
    img_features_norm = F.normalize(img_features, p=2, dim=-1)
    return img_features_norm @ img_features_norm.T


def detect_ood_tokens(
    features: torch.Tensor,
    reference_mean: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """
    Phát hiện các token OOD bằng Cosine Similarity so với phân phối tham chiếu.

    Ý tưởng (từ Đánh_giá_1.docx):
    - Nhóm In-Distribution (xe hơi, xe tải): CosSim > 0.8
    - Nhóm OOD (xe ba gác, chợ tự phát): CosSim thấp, vector bị "trôi dạt"

    Args:
        features:       [N_tokens, D] — token features 1 ảnh
        reference_mean: [D] — vector đặc trưng trung bình tham chiếu (pre-trained distribution)
        threshold:      ngưỡng dưới CosSim, dưới ngưỡng → OOD
    Returns:
        dict với keys:
        - "cosine_scores": [N_tokens] — điểm cosine mỗi token
        - "ood_mask":      [N_tokens] — True nếu token bị OOD
        - "ood_ratio":     float — tỷ lệ token OOD
    """
    features_norm = F.normalize(features, p=2, dim=-1)    # [N, D]
    ref_norm = F.normalize(reference_mean.unsqueeze(0), p=2, dim=-1)  # [1, D]

    cosine_scores = (features_norm @ ref_norm.T).squeeze(-1)  # [N]
    ood_mask = cosine_scores < threshold

    return {
        "cosine_scores": cosine_scores,
        "ood_mask": ood_mask,
        "ood_ratio": ood_mask.float().mean().item(),
    }


def run_cosine_analysis(features_path: str, output_dir: str):
    """Pipeline chính: load features → tính CosSim → xuất kết quả."""

    os.makedirs(output_dir, exist_ok=True)
    data = torch.load(features_path, map_location="cpu")
    features = data["features"]     # [num_images, N_tokens, D]
    filenames = data["filenames"]
    num_images, N, D = features.shape
    print(f"[INFO] Loaded {num_images} images, {N} tokens/image, D={D}")

    # ---- 1. Tính reference distribution (trung bình toàn bộ dataset) ----
    global_mean = features.mean(dim=(0, 1))  # [D]
    print(f"[INFO] Global mean feature norm: {global_mean.norm():.4f}")

    # ---- 2. Tính Cosine Similarity inter-image ----
    inter_sim = compute_inter_image_cosine(features)
    plt.figure(figsize=(10, 8))
    plt.imshow(inter_sim.numpy(), cmap="RdYlGn", vmin=0.5, vmax=1.0)
    plt.colorbar(label="Cosine Similarity")
    plt.title("Inter-Image Cosine Similarity Matrix\n(Mean-pooled token features)")
    plt.xlabel("Image Index")
    plt.ylabel("Image Index")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/inter_image_cosine_matrix.png", dpi=150)
    plt.close()
    print(f"[SAVED] inter_image_cosine_matrix.png")

    # ---- 3. Tính OOD scores cho từng ảnh ----
    ood_results = []
    for i in range(num_images):
        result = detect_ood_tokens(features[i], global_mean, threshold=0.5)
        ood_results.append({
            "filename": filenames[i],
            "mean_cosine": result["cosine_scores"].mean().item(),
            "min_cosine": result["cosine_scores"].min().item(),
            "ood_ratio": result["ood_ratio"],
            "cosine_scores": result["cosine_scores"],
        })

    # ---- 4. Xuất bảng OOD scores ----
    print("\n" + "=" * 80)
    print(f"{'Filename':<35} {'Mean CosSim':>12} {'Min CosSim':>12} {'OOD Ratio':>12}")
    print("=" * 80)
    for r in sorted(ood_results, key=lambda x: x["mean_cosine"]):
        flag = " ⚠️ OOD" if r["ood_ratio"] > 0.3 else ""
        print(f"{r['filename']:<35} {r['mean_cosine']:>12.4f} {r['min_cosine']:>12.4f} {r['ood_ratio']:>12.2%}{flag}")
    print("=" * 80)

    # ---- 5. Vẽ biểu đồ phân phối OOD scores ----
    mean_scores = [r["mean_cosine"] for r in ood_results]
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(mean_scores)), mean_scores, color=["red" if s < 0.6 else "green" for s in mean_scores])
    plt.axhline(y=0.6, color="orange", linestyle="--", label="OOD Threshold (0.6)")
    plt.xlabel("Image Index")
    plt.ylabel("Mean Cosine Similarity")
    plt.title("OOD Score per Image (vs. Global Mean Feature)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ood_scores_bar.png", dpi=150)
    plt.close()
    print(f"[SAVED] ood_scores_bar.png")

    # ---- 6. Lưu kết quả dạng .pt ----
    torch.save(ood_results, f"{output_dir}/ood_results.pt")
    print(f"[SAVED] ood_results.pt")

    # ---- 7. Vẽ 1 mẫu intra-image cosine matrix (ảnh đầu tiên) ----
    intra_sim = compute_cosine_similarity_matrix(features[0])
    plt.figure(figsize=(10, 8))
    plt.imshow(intra_sim.numpy(), cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Cosine Similarity")
    plt.title(f"Intra-Image Token Cosine Similarity\n{filenames[0]}")
    plt.xlabel("Token Index")
    plt.ylabel("Token Index")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/intra_image_cosine_sample.png", dpi=150)
    plt.close()
    print(f"[SAVED] intra_image_cosine_sample.png")


# ======================== CHẠY ========================
if __name__ == "__main__":
    run_cosine_analysis(
        features_path=f"{OUTPUT_DIR}/vf_token_features.pt",
        output_dir=OUTPUT_DIR,
    )
```

---

## 5. Script 3: Mahalanobis Distance

> **Mục đích:** Khắc phục hạn chế của CosSim bằng khoảng cách Mahalanobis.
> Công thức (từ Đánh_giá_1.docx):
>
> $D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$
>
> Trong đó: `μ` là vector kỳ vọng, `Σ` là ma trận hiệp phương sai.

```python
#!/usr/bin/env python3
"""
Script 3: Tính Mahalanobis Distance cho OOD Detection.
Chuẩn hóa không gian ẩn bằng ma trận hiệp phương sai, tự động "phạt"
các chiều nhiễu và làm nổi bật sự dị biệt thực sự.
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = "/content/vinfast"
OUTPUT_DIR = f"{WORKSPACE}/outputs/spatial_eval"


def compute_mahalanobis_distance(
    features: torch.Tensor,
    mean: torch.Tensor,
    cov_inv: torch.Tensor,
) -> torch.Tensor:
    """
    Tính khoảng cách Mahalanobis cho mỗi token.

    D_M(x) = sqrt( (x - μ)ᵀ Σ⁻¹ (x - μ) )

    Args:
        features: [N, D] — token features
        mean:     [D]    — vector kỳ vọng μ
        cov_inv:  [D, D] — ma trận hiệp phương sai nghịch đảo Σ⁻¹
    Returns:
        distances: [N] — Mahalanobis distance cho mỗi token
    """
    diff = features - mean.unsqueeze(0)          # [N, D]
    left = diff @ cov_inv                         # [N, D]
    mahal_sq = (left * diff).sum(dim=-1)          # [N]
    # Clamp để tránh giá trị âm do lỗi số (numerical error)
    mahal_sq = torch.clamp(mahal_sq, min=0.0)
    return torch.sqrt(mahal_sq)                   # [N]


def fit_reference_distribution(features: torch.Tensor, shrinkage: float = 0.01):
    """
    Ước lượng phân phối tham chiếu (μ, Σ) từ toàn bộ token features.

    Sử dụng Ledoit-Wolf shrinkage để ổn định ma trận hiệp phương sai
    khi D >> N (nghĩa là chiều feature lớn hơn số mẫu).

    Args:
        features:  [total_tokens, D] — tất cả token features gộp lại
        shrinkage: hệ số co (shrinkage), 0.01 = nhẹ, 0.1 = mạnh
    Returns:
        mean:    [D]
        cov_inv: [D, D]
    """
    D = features.shape[1]

    # Tính μ (mean vector)
    mean = features.mean(dim=0)  # [D]

    # Tính Σ (covariance matrix)
    centered = features - mean.unsqueeze(0)
    cov = (centered.T @ centered) / (centered.shape[0] - 1)  # [D, D]

    # Ledoit-Wolf shrinkage: Σ_reg = (1 - α)·Σ + α·trace(Σ)/D · I
    trace_cov = torch.trace(cov)
    cov_reg = (1 - shrinkage) * cov + shrinkage * (trace_cov / D) * torch.eye(D)

    # Tính Σ⁻¹
    cov_inv = torch.linalg.inv(cov_reg)

    return mean, cov_inv


def run_mahalanobis_analysis(features_path: str, output_dir: str):
    """Pipeline chính: load features → fit distribution → tính Mahalanobis → xuất."""

    os.makedirs(output_dir, exist_ok=True)
    data = torch.load(features_path, map_location="cpu")
    features = data["features"]     # [num_images, N_tokens, D]
    filenames = data["filenames"]
    num_images, N, D = features.shape
    print(f"[INFO] Loaded {num_images} images, {N} tokens/image, D={D}")

    # ---- 1. Fit phân phối tham chiếu trên toàn bộ token ----
    all_tokens = features.reshape(-1, D)  # [num_images * N, D]
    print(f"[INFO] Fitting reference distribution on {all_tokens.shape[0]} tokens...")

    # Dùng float64 cho ổn định số (numerical stability)
    mean, cov_inv = fit_reference_distribution(all_tokens.double())
    mean = mean.float()
    cov_inv = cov_inv.float()
    print(f"[INFO] Reference μ norm: {mean.norm():.4f}")

    # ---- 2. Tính Mahalanobis Distance cho từng ảnh ----
    mahal_results = []
    for i in range(num_images):
        distances = compute_mahalanobis_distance(features[i], mean, cov_inv)
        mahal_results.append({
            "filename": filenames[i],
            "mean_mahal": distances.mean().item(),
            "max_mahal": distances.max().item(),
            "std_mahal": distances.std().item(),
            "distances": distances,   # [N_tokens]
        })

    # ---- 3. Xuất bảng kết quả ----
    print("\n" + "=" * 90)
    print(f"{'Filename':<35} {'Mean D_M':>12} {'Max D_M':>12} {'Std D_M':>12} {'OOD?':>8}")
    print("=" * 90)

    # Tính ngưỡng OOD = mean + 2*std của toàn bộ mean_mahal
    all_means = [r["mean_mahal"] for r in mahal_results]
    threshold = np.mean(all_means) + 2 * np.std(all_means)

    for r in sorted(mahal_results, key=lambda x: x["mean_mahal"], reverse=True):
        flag = "⚠️ YES" if r["mean_mahal"] > threshold else ""
        print(f"{r['filename']:<35} {r['mean_mahal']:>12.2f} {r['max_mahal']:>12.2f} {r['std_mahal']:>12.2f} {flag:>8}")
    print("=" * 90)
    print(f"[INFO] OOD Threshold (μ + 2σ): {threshold:.2f}")

    # ---- 4. Vẽ biểu đồ phân phối Mahalanobis ----
    plt.figure(figsize=(12, 5))
    colors = ["red" if m > threshold else "steelblue" for m in all_means]
    plt.bar(range(len(all_means)), all_means, color=colors)
    plt.axhline(y=threshold, color="orange", linestyle="--", label=f"OOD Threshold ({threshold:.1f})")
    plt.xlabel("Image Index")
    plt.ylabel("Mean Mahalanobis Distance")
    plt.title("Mahalanobis OOD Score per Image")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mahalanobis_scores_bar.png", dpi=150)
    plt.close()
    print(f"[SAVED] mahalanobis_scores_bar.png")

    # ---- 5. Vẽ histogram phân phối token-level Mahalanobis ----
    all_dists = torch.cat([r["distances"] for r in mahal_results]).numpy()
    plt.figure(figsize=(10, 5))
    plt.hist(all_dists, bins=100, color="steelblue", edgecolor="white", alpha=0.8)
    plt.axvline(x=np.percentile(all_dists, 95), color="red", linestyle="--",
                label=f"95th percentile ({np.percentile(all_dists, 95):.1f})")
    plt.xlabel("Mahalanobis Distance")
    plt.ylabel("Number of Tokens")
    plt.title("Distribution of Token-Level Mahalanobis Distance\n(All VinFast images)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mahalanobis_histogram.png", dpi=150)
    plt.close()
    print(f"[SAVED] mahalanobis_histogram.png")

    # ---- 6. Lưu kết quả ----
    torch.save({
        "mahal_results": mahal_results,
        "reference_mean": mean,
        "reference_cov_inv": cov_inv,
        "ood_threshold": threshold,
    }, f"{output_dir}/mahalanobis_results.pt")
    print(f"[SAVED] mahalanobis_results.pt")


# ======================== CHẠY ========================
if __name__ == "__main__":
    run_mahalanobis_analysis(
        features_path=f"{OUTPUT_DIR}/vf_token_features.pt",
        output_dir=OUTPUT_DIR,
    )
```

---

## 6. Script 4: Spatial Attention Heatmaps

> **Mục đích:** Trích xuất trọng số Attention từ block cuối cùng của ViT-Large,
> reshape thành lưới 2D, overlay lên ảnh gốc dưới dạng bản đồ nhiệt (heatmap).
> Đồng thời overlay cả Cosine OOD scores thành heatmap thứ hai.

```python
#!/usr/bin/env python3
"""
Script 4: Tạo Spatial Attention Heatmaps và OOD Heatmaps chồng lên ảnh gốc.
Cần hook vào forward pass của encoder để trích xuất attention weights.
"""
import sys
import os

WORKSPACE = "/content/vinfast"
VF_DRIVE_JEPA = f"{WORKSPACE}/src/vf-drive-jepa"
sys.path.insert(0, VF_DRIVE_JEPA)
sys.path.insert(0, f"{VF_DRIVE_JEPA}/navsim")
sys.path.insert(0, f"{VF_DRIVE_JEPA}/vjepa2")
os.chdir(VF_DRIVE_JEPA)

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
import yaml
import cv2

import src.models.vision_transformer as vit


# ---- Biến global để hook ghi lại attention weights ----
_attention_store = {}


def _attention_hook_fn(module, input, output, layer_name):
    """
    Hook function để bắt attention weights từ các block ViT.
    Vì ViT dùng SDPA (scaled_dot_product_attention) mặc định,
    ta cần tắt SDPA và dùng manual attention để bắt được weights.
    """
    pass  # Xem phương pháp thay thế bên dưới


def load_encoder_with_attention(
    checkpoint_path: str,
    resolution=(256, 512),
    config_path: str = "./vjepa2/configs/eval/vitl/in1k.yaml",
):
    """
    Load encoder ViT-Large và cấu hình để trích xuất attention weights.
    Trick: Tắt use_sdpa để attention weights được tính tường minh.
    """
    with open(config_path, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    model_kwargs = params["model_kwargs"]
    wrapper_kwargs = model_kwargs["wrapper_kwargs"]
    wrapper_kwargs["img_as_video_nframes"] = 2
    model_kwargs = model_kwargs["pretrain_kwargs"]
    model_kwargs["encoder"]["model_name"] = "vit_large"
    # Tắt SDPA để có thể trích xuất attention weights
    model_kwargs["encoder"]["use_sdpa"] = False

    # Load checkpoint thủ công (không dùng init_module để kiểm soát nhiều hơn)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    enc_kwargs = model_kwargs["encoder"]
    enc_ckp_key = enc_kwargs.get("checkpoint_key")
    enc_model_name = enc_kwargs.get("model_name")

    model = vit.__dict__[enc_model_name](
        input_size=resolution,
        num_frames=wrapper_kwargs["img_as_video_nframes"],
        **enc_kwargs,
    )

    # Đăng ký prehook giống init_module
    nframes = wrapper_kwargs["img_as_video_nframes"]
    def forward_prehook(module, input):
        input = input[0]
        input = input.unsqueeze(2).repeat(1, 1, nframes, 1, 1)
        return input
    model.register_forward_pre_hook(forward_prehook)

    # Load weights
    pretrained_dict = ckpt[enc_ckp_key]
    pretrained_dict = {k.replace("module.", "").replace("backbone.", ""): v
                       for k, v in pretrained_dict.items()}
    for k, v in model.state_dict().items():
        if k not in pretrained_dict:
            pass
        elif pretrained_dict[k].shape != v.shape:
            pretrained_dict[k] = v
    model.load_state_dict(pretrained_dict, strict=False)
    model.eval()
    del ckpt
    return model


@torch.no_grad()
def extract_attention_weights(
    model,
    img_tensor: torch.Tensor,
    device: str = "cuda",
) -> np.ndarray:
    """
    Trích xuất attention weights từ block cuối cùng của ViT.

    Cách hoạt động:
    1. Forward pass qua tất cả block trừ block cuối
    2. Tại block cuối, tính Q, K thủ công để lấy attention map

    Returns:
        attn_map: [H_patches, W_patches] — trung bình attention per spatial token
    """
    model = model.to(device)
    img_tensor = img_tensor.to(device)

    # Forward qua patch embedding + pos embedding
    x = img_tensor
    if x.ndim == 4:
        _, _, H, W = x.shape
        T = 1
    elif x.ndim == 5:
        _, _, T, H, W = x.shape
        T = T // model.tubelet_size

    H_patches = H // model.patch_size
    W_patches = W // model.patch_size

    if not model.use_rope:
        pos_embed = model.interpolate_pos_encoding(x, model.pos_embed)
        x = model.patch_embed(x)
        x += pos_embed
    else:
        x = model.patch_embed(x)

    # Forward qua tất cả block trừ block cuối
    for blk in model.blocks[:-1]:
        x = blk(x, T=T, H_patches=H_patches, W_patches=W_patches)

    # Tại block cuối: tính attention map thủ công
    last_block = model.blocks[-1]
    y = last_block.norm1(x)

    attn_module = last_block.attn
    B, N, C = y.size()

    if hasattr(attn_module, 'qkv'):
        qkv = attn_module.qkv(y)
        num_heads = attn_module.num_heads
        head_dim = C // num_heads
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        scale = head_dim ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale  # [B, heads, N, N]
        attn_weights = attn_weights.softmax(dim=-1)

        # Lấy trung bình qua tất cả heads và batch
        attn_mean = attn_weights.mean(dim=(0, 1))  # [N, N]

        # Lấy attention từ mỗi token tới tất cả token khác → tổng hợp
        # Sử dụng cột trung bình (mỗi token nhận bao nhiêu attention)
        attn_per_token = attn_mean.mean(dim=0)  # [N]

        # Reshape về lưới 2D không gian
        # N = T_eff × H_patches × W_patches, với T_eff=1 (do tubelet_size=2, num_frames=2)
        T_eff = max(1, T)
        spatial_N = H_patches * W_patches
        if attn_per_token.shape[0] == T_eff * spatial_N:
            attn_spatial = attn_per_token[:spatial_N]  # lấy frame đầu
        else:
            attn_spatial = attn_per_token[:spatial_N]

        attn_map = attn_spatial.cpu().numpy().reshape(H_patches, W_patches)
        return attn_map
    else:
        raise RuntimeError("Attention module không có thuộc tính 'qkv'")


def create_attention_heatmap(
    image_path: str,
    attn_map: np.ndarray,
    output_path: str,
    title: str = "Spatial Attention Heatmap",
):
    """
    Overlay bản đồ nhiệt attention lên ảnh gốc.

    Args:
        image_path: đường dẫn ảnh gốc (.jpg)
        attn_map: [H_patches, W_patches] — attention weights
        output_path: đường dẫn lưu ảnh kết quả
        title: tiêu đề
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]

    # Resize attention map lên kích thước ảnh gốc
    attn_resized = cv2.resize(attn_map, (W, H), interpolation=cv2.INTER_CUBIC)

    # Chuẩn hóa về [0, 1]
    attn_min, attn_max = attn_resized.min(), attn_resized.max()
    if attn_max - attn_min > 1e-8:
        attn_norm = (attn_resized - attn_min) / (attn_max - attn_min)
    else:
        attn_norm = np.zeros_like(attn_resized)

    # Tạo heatmap màu
    heatmap = plt.cm.jet(attn_norm)[:, :, :3]  # [H, W, 3] ∈ [0, 1]
    heatmap = (heatmap * 255).astype(np.uint8)

    # Overlay: ảnh gốc 60% + heatmap 40%
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].imshow(img)
    axes[0].set_title("Ảnh gốc")
    axes[0].axis("off")

    axes[1].imshow(attn_norm, cmap="jet")
    axes[1].set_title("Attention Map (Raw)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(title)
    axes[2].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_ood_heatmap(
    image_path: str,
    cosine_scores: torch.Tensor,
    H_patches: int,
    W_patches: int,
    output_path: str,
):
    """
    Overlay bản đồ nhiệt OOD (dựa trên Cosine Similarity) lên ảnh gốc.
    Vùng có CosSim thấp (OOD) sẽ hiện màu đỏ nóng.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]

    # Reshape cosine scores thành lưới 2D
    ood_map = (1.0 - cosine_scores[:H_patches * W_patches]).numpy().reshape(H_patches, W_patches)
    ood_resized = cv2.resize(ood_map, (W, H), interpolation=cv2.INTER_CUBIC)

    # Chuẩn hóa
    ood_min, ood_max = ood_resized.min(), ood_resized.max()
    if ood_max - ood_min > 1e-8:
        ood_norm = (ood_resized - ood_min) / (ood_max - ood_min)
    else:
        ood_norm = np.zeros_like(ood_resized)

    heatmap = plt.cm.hot(ood_norm)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(img)
    axes[0].set_title("Ảnh gốc")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("OOD Heatmap (Cosine-based)\nĐỎ = Khả năng OOD cao")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_heatmap_pipeline(
    checkpoint_path: str,
    camera_dir: str,
    features_path: str,
    output_dir: str,
    max_images: int = 10,
):
    """Pipeline chính: tạo attention heatmaps và OOD heatmaps."""

    os.makedirs(f"{output_dir}/heatmaps", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load encoder (tắt SDPA)
    print("[INFO] Loading encoder (SDPA disabled for attention extraction)...")
    encoder = load_encoder_with_attention(checkpoint_path)

    # Load OOD results nếu có
    ood_data = None
    if os.path.exists(features_path):
        data = torch.load(features_path, map_location="cpu")
        features = data["features"]
        filenames = data["filenames"]
        global_mean = features.mean(dim=(0, 1))
    else:
        features = None
        filenames = None

    image_transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    image_paths = sorted(Path(camera_dir).glob("*.jpg"))[:max_images]
    H_patches, W_patches = 16, 32  # resolution (256,512), patch_size=16

    for i, img_path in enumerate(image_paths):
        print(f"[INFO] Processing {i+1}/{len(image_paths)}: {img_path.name}")

        # 1. Tạo Attention Heatmap
        img_tensor = image_transform(Image.open(img_path).convert("RGB")).unsqueeze(0)
        attn_map = extract_attention_weights(encoder, img_tensor, device)

        attn_output = f"{output_dir}/heatmaps/attention_{img_path.stem}.png"
        create_attention_heatmap(
            str(img_path), attn_map, attn_output,
            title=f"Spatial Attention — {img_path.name}"
        )

        # 2. Tạo OOD Heatmap (nếu có features)
        if features is not None and img_path.name in filenames:
            idx = filenames.index(img_path.name)
            result = {}
            feat_norm = F.normalize(features[idx], p=2, dim=-1)
            ref_norm = F.normalize(global_mean.unsqueeze(0), p=2, dim=-1)
            cosine_scores = (feat_norm @ ref_norm.T).squeeze(-1)

            ood_output = f"{output_dir}/heatmaps/ood_{img_path.stem}.png"
            create_ood_heatmap(str(img_path), cosine_scores, H_patches, W_patches, ood_output)

    print(f"[DONE] Heatmaps saved to {output_dir}/heatmaps/")


# ======================== CHẠY ========================
if __name__ == "__main__":
    ENCODER_CKPT = f"{WORKSPACE}/.cache/checkpoints/vjepa2/vitl_merge_3dataset_e50.pt"
    CAMERA_DIR = f"{WORKSPACE}/dataset_vf/data/CAMERA/CAM_P_F"
    FEATURES_PATH = f"{WORKSPACE}/outputs/spatial_eval/vf_token_features.pt"
    OUTPUT_DIR = f"{WORKSPACE}/outputs/spatial_eval"

    run_heatmap_pipeline(
        checkpoint_path=ENCODER_CKPT,
        camera_dir=CAMERA_DIR,
        features_path=FEATURES_PATH,
        output_dir=OUTPUT_DIR,
        max_images=10,
    )
```

---

## 7. Script 5: PCA Visualization

> **Ý tưởng (từ Slide):**
> Áp dụng thuật toán PCA lên các feature patches để giảm chiều không gian ẩn
> xuống dạng 2D/3D (giống biểu đồ heatmap). Phủ bản đồ này lên khung hình gốc
> để thấy sự sai lệch màu sắc tại các vùng có xe tự chế, xe ba gác, hạ tầng lạ.
>
> **Công thức:**
> - Token features: `z ∈ R^{N×1024}` (N=512 tokens, D=1024)
> - PCA → giảm xuống 3 thành phần chính: `z_pca ∈ R^{N×3}`
> - Reshape thành lưới không gian: `[H_patches × W_patches × 3]`
> - Chuẩn hóa RGB → overlay lên ảnh gốc

Paste vào **1 ô code** trên Colab:

```python
#!/usr/bin/env python3
"""
Script 5: PCA Visualization — Trực quan hóa không gian đặc trưng bằng PCA.
Giảm chiều 1024D → 3D (RGB), reshape thành bản đồ không gian, overlay lên ảnh gốc.

Input:  outputs/spatial_eval/vf_token_features.pt (từ Script 1)
Output: outputs/spatial_eval/pca/*.png
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import cv2

WORKSPACE = "/content/vinfast"
OUTPUT_DIR = f"{WORKSPACE}/outputs/spatial_eval"


def pca_reduce(features: torch.Tensor, n_components: int = 3) -> torch.Tensor:
    """
    PCA giảm chiều bằng SVD (không cần sklearn).

    Công thức:
        1. Trung tâm hóa: X_c = X - mean(X)
        2. SVD:            X_c = U · S · Vᵀ
        3. Chiếu:          Z_pca = X_c · V[:, :k]   (k = n_components)

    Args:
        features: [N, D] — token features (D=1024)
        n_components: số chiều PCA (3 cho RGB)
    Returns:
        projected: [N, n_components]
    """
    # 1. Trung tâm hóa (zero-mean)
    mean = features.mean(dim=0, keepdim=True)
    X_centered = features - mean  # [N, D]

    # 2. SVD (dùng torch.linalg.svd, kinh tế hơn full SVD)
    # full_matrices=False: chỉ tính min(N, D) singular vectors
    U, S, Vt = torch.linalg.svd(X_centered.float(), full_matrices=False)

    # 3. Chiếu lên k thành phần chính đầu tiên
    # V[:, :k] = Vt[:k, :].T
    V_k = Vt[:n_components, :].T  # [D, k]
    projected = X_centered.float() @ V_k  # [N, k]

    # 4. Tính phương sai giải thích (explained variance ratio)
    total_var = (S ** 2).sum()
    explained_var = (S[:n_components] ** 2) / total_var
    print(f"    PCA Explained Variance Ratio: {explained_var.numpy()}")
    print(f"    Tổng phương sai giải thích:   {explained_var.sum():.2%}")

    return projected


def normalize_to_rgb(pca_map: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa mỗi kênh PCA về [0, 255] để hiển thị dạng RGB.
    Dùng min-max scaling riêng cho từng kênh.
    """
    rgb = np.zeros_like(pca_map, dtype=np.float32)
    for c in range(3):
        channel = pca_map[:, :, c]
        c_min, c_max = channel.min(), channel.max()
        if c_max - c_min > 1e-8:
            rgb[:, :, c] = (channel - c_min) / (c_max - c_min)
        else:
            rgb[:, :, c] = 0.5
    return (rgb * 255).astype(np.uint8)


def create_pca_visualization(
    image_path: str,
    pca_features: np.ndarray,
    H_patches: int,
    W_patches: int,
    output_path: str,
    title: str = "",
):
    """
    Tạo ảnh PCA overlay: reshape PCA 3-component thành RGB map,
    resize lên kích thước ảnh gốc, overlay lên.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]

    # Reshape thành lưới không gian [H_patches, W_patches, 3]
    pca_map = pca_features[:H_patches * W_patches].reshape(H_patches, W_patches, 3)
    pca_rgb = normalize_to_rgb(pca_map)

    # Resize lên kích thước ảnh gốc (bilinear interpolation)
    pca_resized = cv2.resize(pca_rgb, (W, H), interpolation=cv2.INTER_CUBIC)

    # Overlay: ảnh gốc 50% + PCA map 50%
    overlay = cv2.addWeighted(img, 0.5, pca_resized, 0.5, 0)

    # Vẽ kết quả
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    axes[0].imshow(img)
    axes[0].set_title("Ảnh gốc (Camera VinFast)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(pca_rgb)
    axes[1].set_title("PCA Feature Map (RGB)\nMỗi màu = 1 thành phần chính", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("PCA Overlay\nVùng khác màu = đặc trưng khác biệt", fontsize=12)
    axes[2].axis("off")

    plt.suptitle(title or "PCA Visualization — Phân Tích Không Gian Đặc Trưng",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_pca_global_scatter(
    all_pca: np.ndarray,
    all_labels: list,
    output_path: str,
):
    """
    Vẽ scatter plot 2D (PC1 vs PC2) của tất cả token từ tất cả ảnh.
    Mỗi ảnh một màu → trực quan hóa domain gap giữa các cảnh.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    unique_labels = list(set(all_labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))

    for idx, label in enumerate(unique_labels):
        mask = [l == label for l in all_labels]
        points = all_pca[mask]
        ax.scatter(points[:, 0], points[:, 1],
                   c=[cmap(idx)], s=3, alpha=0.4, label=label[:20])

    ax.set_xlabel("PC1 (Thành phần chính 1)", fontsize=12)
    ax.set_ylabel("PC2 (Thành phần chính 2)", fontsize=12)
    ax.set_title("PCA Global Scatter — Phân Phối Token Trong Không Gian Ẩn\n"
                 "(Mỗi màu = 1 ảnh, cụm xa = đặc trưng khác biệt lớn)", fontsize=13)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_pca_pipeline(features_path: str, camera_dir: str, output_dir: str, max_images: int = 10):
    """Pipeline chính: load features → PCA → vẽ."""

    pca_dir = f"{output_dir}/pca"
    os.makedirs(pca_dir, exist_ok=True)

    data = torch.load(features_path, map_location="cpu")
    features = data["features"]     # [num_images, N_tokens, D]
    filenames = data["filenames"]
    num_images, N, D = features.shape
    print(f"[INFO] Loaded {num_images} images, {N} tokens/image, D={D}")

    H_patches, W_patches = 16, 32  # resolution (256, 512), patch_size=16

    # ---- 1. PCA toàn cục: gộp tất cả token, giảm 1024D → 3D ----
    print("[INFO] Running PCA on all tokens (1024D → 3D)...")
    all_tokens = features.reshape(-1, D)  # [num_images * N, D]
    pca_all = pca_reduce(all_tokens, n_components=3)  # [total_tokens, 3]
    pca_all_np = pca_all.numpy()

    # ---- 2. Vẽ PCA overlay cho từng ảnh ----
    from pathlib import Path
    image_paths = sorted(Path(camera_dir).glob("*.jpg"))
    # Tạo dict để tìm nhanh
    img_map = {p.name: str(p) for p in image_paths}

    n_to_plot = min(max_images, num_images)
    for i in range(n_to_plot):
        fname = filenames[i]
        if fname not in img_map:
            continue
        print(f"[INFO] PCA viz {i+1}/{n_to_plot}: {fname}")

        # Lấy PCA features của ảnh thứ i
        pca_img = pca_all_np[i * N : (i + 1) * N]  # [N, 3]

        output_path = f"{pca_dir}/pca_{Path(fname).stem}.png"
        create_pca_visualization(
            image_path=img_map[fname],
            pca_features=pca_img,
            H_patches=H_patches,
            W_patches=W_patches,
            output_path=output_path,
            title=f"PCA Feature Map — {fname}",
        )

    # ---- 3. Vẽ scatter plot toàn cục ----
    print("[INFO] Creating global PCA scatter plot...")
    labels = []
    for i, fname in enumerate(filenames):
        labels.extend([fname] * N)

    create_pca_global_scatter(
        all_pca=pca_all_np,
        all_labels=labels,
        output_path=f"{pca_dir}/pca_global_scatter.png",
    )

    print(f"[DONE] PCA visualizations saved to {pca_dir}/")


# ======================== CHẠY ========================
if __name__ == "__main__":
    FEATURES_PATH = f"{OUTPUT_DIR}/vf_token_features.pt"
    CAMERA_DIR = f"{WORKSPACE}/dataset_vf/data/CAMERA/CAM_P_F"

    run_pca_pipeline(
        features_path=FEATURES_PATH,
        camera_dir=CAMERA_DIR,
        output_dir=OUTPUT_DIR,
        max_images=10,
    )
```

---

## 8. Script 6: Occlusion Mapping

> **Ý tưởng (từ Slide):**
> Đánh giá sự thay đổi biểu diễn của mô hình khi một vùng ảnh `Ω ⊂ R^{H×W}`
> bị che khuất. Mục tiêu là kiểm tra khả năng dự đoán đặc trưng của các vật thể
> dị hình trong môi trường giao thông đông đúc.
>
> Nếu bản đồ che khuất cho thấy độ nhạy tập trung chủ yếu ở vùng trung tâm đường
> và bỏ qua các đối tượng nhỏ ở biên ảnh, điều đó là minh chứng cho domain gap
> ở mức không gian biểu diễn.
>
> **Công thức:**
> - Feature gốc (không che): `z₀ = Encoder(I)`
> - Feature khi che vùng (r,c): `z_{r,c} = Encoder(I ⊙ M_{r,c})`
>   (M là mask che khuất tại vị trí lưới r,c)
> - Sensitivity = `||z₀ - z_{r,c}||₂` (L2 norm chênh lệch)
> - Sensitivity Map: `S[r, c]` — càng lớn = che vùng đó ảnh hưởng càng nhiều

Paste vào **1 ô code** trên Colab:

```python
#!/usr/bin/env python3
"""
Script 6: Occlusion Mapping — Đo độ nhạy không gian bằng kỹ thuật che khuất.
Trượt một vùng đen qua ảnh, so sánh feature thay đổi bao nhiêu.

Input:  Encoder checkpoint + ảnh camera
Output: outputs/spatial_eval/occlusion/*.png
"""
import sys
import os

WORKSPACE = "/content/vinfast"
VF_DRIVE_JEPA = f"{WORKSPACE}/src/vf-drive-jepa"
sys.path.insert(0, VF_DRIVE_JEPA)
sys.path.insert(0, f"{VF_DRIVE_JEPA}/navsim")
sys.path.insert(0, f"{VF_DRIVE_JEPA}/vjepa2")
os.chdir(VF_DRIVE_JEPA)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
import cv2
import yaml

from vjepa2.evals.image_classification_frozen.modelcustom.vit_encoder import init_module


def load_encoder(checkpoint_path, resolution=(256, 512)):
    """Load encoder V-JEPA2 (giống Script 1)."""
    config_path = f"{VF_DRIVE_JEPA}/vjepa2/configs/eval/vitl/in1k.yaml"
    with open(config_path, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    model_kwargs = params["model_kwargs"]
    wrapper_kwargs = model_kwargs["wrapper_kwargs"]
    wrapper_kwargs["img_as_video_nframes"] = 2
    model_kwargs = model_kwargs["pretrain_kwargs"]
    model_kwargs["encoder"]["model_name"] = "vit_large"

    encoder = init_module(
        resolution=resolution,
        checkpoint=checkpoint_path,
        model_kwargs=model_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        register_prehook=True,
    )
    encoder.eval()
    return encoder


# Tiền xử lý ảnh
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


@torch.no_grad()
def compute_occlusion_sensitivity(
    encoder: torch.nn.Module,
    image_path: str,
    device: str = "cuda",
    occlusion_size: int = 32,
    stride: int = 16,
) -> dict:
    """
    Tính Occlusion Sensitivity Map.

    Thuật toán:
    1. Forward pass ảnh gốc → lấy feature baseline z₀
    2. Lặp qua mọi vị trí (r, c) trên ảnh:
       a. Tạo bản sao ảnh, tô đen vùng [r:r+S, c:c+S]
       b. Forward pass ảnh bị che → z_{r,c}
       c. Sensitivity[r, c] = ||z₀ - z_{r,c}||₂
    3. Reshape → lưới sensitivity map

    Args:
        encoder:        V-JEPA2 encoder
        image_path:     đường dẫn ảnh .jpg
        occlusion_size: kích thước ô che (pixels), mặc định 32 = 2 patches
        stride:         bước nhảy khi trượt, mặc định 16 = 1 patch
    Returns:
        dict với keys:
        - "sensitivity_map": [H_grid, W_grid] — bản đồ độ nhạy
        - "max_sensitivity_pos": (row, col) — vị trí nhạy nhất
        - "min_sensitivity_pos": (row, col) — vị trí ít nhạy nhất
    """
    encoder = encoder.to(device)

    # 1. Ảnh gốc → feature baseline
    img = Image.open(image_path).convert("RGB")
    img_tensor = IMAGE_TRANSFORM(img).unsqueeze(0).to(device)  # [1, 3, 256, 512]

    baseline_features = encoder(img_tensor)  # [1, N, D]
    baseline_flat = baseline_features.reshape(-1)  # [N*D]

    _, _, H, W = img_tensor.shape  # 256, 512

    # 2. Tính toán grid
    rows = list(range(0, H - occlusion_size + 1, stride))
    cols = list(range(0, W - occlusion_size + 1, stride))
    sensitivity_map = np.zeros((len(rows), len(cols)), dtype=np.float32)

    total = len(rows) * len(cols)
    count = 0

    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            # Tạo bản sao và che khuất
            occluded = img_tensor.clone()
            occluded[:, :, r:r + occlusion_size, c:c + occlusion_size] = 0.0

            # Forward
            occ_features = encoder(occluded)  # [1, N, D]
            occ_flat = occ_features.reshape(-1)

            # L2 distance (sensitivity)
            dist = torch.norm(baseline_flat - occ_flat, p=2).item()
            sensitivity_map[ri, ci] = dist

            count += 1

        if (ri + 1) % 4 == 0:
            print(f"    Occlusion progress: {count}/{total} ({100*count/total:.0f}%)")

    # 3. Tìm vị trí nhạy nhất / ít nhạy nhất
    max_idx = np.unravel_index(sensitivity_map.argmax(), sensitivity_map.shape)
    min_idx = np.unravel_index(sensitivity_map.argmin(), sensitivity_map.shape)

    return {
        "sensitivity_map": sensitivity_map,
        "max_sensitivity_pos": (rows[max_idx[0]], cols[max_idx[1]]),
        "min_sensitivity_pos": (rows[min_idx[0]], cols[min_idx[1]]),
        "max_sensitivity_val": sensitivity_map.max(),
        "min_sensitivity_val": sensitivity_map.min(),
        "occlusion_size": occlusion_size,
        "stride": stride,
    }


def create_occlusion_visualization(
    image_path: str,
    result: dict,
    output_path: str,
    title: str = "",
):
    """Overlay bản đồ Occlusion Sensitivity lên ảnh gốc."""

    img = np.array(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]

    sensitivity_map = result["sensitivity_map"]

    # Resize sensitivity map lên kích thước ảnh gốc
    sens_resized = cv2.resize(sensitivity_map, (W, H), interpolation=cv2.INTER_CUBIC)

    # Chuẩn hóa [0, 1]
    s_min, s_max = sens_resized.min(), sens_resized.max()
    if s_max - s_min > 1e-8:
        sens_norm = (sens_resized - s_min) / (s_max - s_min)
    else:
        sens_norm = np.zeros_like(sens_resized)

    # Tạo heatmap
    heatmap = plt.cm.hot(sens_norm)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    # Vẽ bounding box tại vị trí nhạy nhất
    max_r, max_c = result["max_sensitivity_pos"]
    occ_size = result["occlusion_size"]
    # Scale vị trí nếu ảnh gốc lớn hơn 256x512
    scale_h = H / 256
    scale_w = W / 512
    max_r_scaled = int(max_r * scale_h)
    max_c_scaled = int(max_c * scale_w)
    occ_h = int(occ_size * scale_h)
    occ_w = int(occ_size * scale_w)
    cv2.rectangle(overlay,
                  (max_c_scaled, max_r_scaled),
                  (max_c_scaled + occ_w, max_r_scaled + occ_h),
                  (0, 255, 0), 3)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    axes[0].imshow(img)
    axes[0].set_title("Ảnh gốc", fontsize=12)
    axes[0].axis("off")

    im = axes[1].imshow(sensitivity_map, cmap="hot", aspect="auto")
    axes[1].set_title(f"Occlusion Sensitivity Map\n(Patch {occ_size}×{occ_size}px)", fontsize=12)
    axes[1].set_xlabel("Column (pixel)")
    axes[1].set_ylabel("Row (pixel)")
    plt.colorbar(im, ax=axes[1], label="L2 Feature Distance")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (□ xanh = vùng nhạy nhất)", fontsize=12)
    axes[2].axis("off")

    plt.suptitle(title or "Occlusion Mapping — Phân Tích Độ Nhạy Không Gian",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_sensitivity_summary(
    all_results: list,
    output_path: str,
):
    """
    Vẽ biểu đồ tổng hợp: trung bình sensitivity theo hàng (row) cho tất cả ảnh.
    Cho thấy mô hình tập trung nhìn vào dải nào của ảnh (trên/giữa/dưới).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Sensitivity trung bình theo hàng (vertical profile)
    all_row_means = []
    for r in all_results:
        smap = r["sensitivity_map"]
        row_mean = smap.mean(axis=1)  # trung bình mỗi hàng
        # Chuẩn hóa
        row_mean = (row_mean - row_mean.min()) / (row_mean.max() - row_mean.min() + 1e-8)
        all_row_means.append(row_mean)

    avg_row_profile = np.mean(all_row_means, axis=0)
    y_labels = np.linspace(0, 256, len(avg_row_profile))

    axes[0].barh(y_labels, avg_row_profile, height=256/len(avg_row_profile), color="coral")
    axes[0].set_ylabel("Vị trí pixel (top → bottom)")
    axes[0].set_xlabel("Normalized Sensitivity")
    axes[0].set_title("Sensitivity Trung Bình Theo Hàng\n(Trên = bầu trời, Dưới = mặt đường)")
    axes[0].invert_yaxis()
    # Thêm annotation
    mid = len(avg_row_profile) // 2
    axes[0].axhline(y=y_labels[mid], color="blue", linestyle="--", alpha=0.5, label="Giữa ảnh")
    axes[0].legend()

    # 2. Sensitivity trung bình theo cột (horizontal profile)
    all_col_means = []
    for r in all_results:
        smap = r["sensitivity_map"]
        col_mean = smap.mean(axis=0)
        col_mean = (col_mean - col_mean.min()) / (col_mean.max() - col_mean.min() + 1e-8)
        all_col_means.append(col_mean)

    avg_col_profile = np.mean(all_col_means, axis=0)
    x_labels = np.linspace(0, 512, len(avg_col_profile))

    axes[1].bar(x_labels, avg_col_profile, width=512/len(avg_col_profile), color="steelblue")
    axes[1].set_xlabel("Vị trí pixel (trái → phải)")
    axes[1].set_ylabel("Normalized Sensitivity")
    axes[1].set_title("Sensitivity Trung Bình Theo Cột\n(Trái/Phải = biên, Giữa = trung tâm)")
    mid_c = len(avg_col_profile) // 2
    axes[1].axvline(x=x_labels[mid_c], color="red", linestyle="--", alpha=0.5, label="Trung tâm")
    axes[1].legend()

    plt.suptitle("Tổng Hợp Occlusion Sensitivity — Mô Hình Tập Trung Nhìn Vào Đâu?",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_occlusion_pipeline(
    checkpoint_path: str,
    camera_dir: str,
    output_dir: str,
    max_images: int = 5,
    occlusion_size: int = 32,
    stride: int = 16,
):
    """Pipeline chính: load encoder → occlusion sweep → vẽ kết quả."""

    occ_dir = f"{output_dir}/occlusion"
    os.makedirs(occ_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # Load encoder
    print("[INFO] Loading encoder for occlusion mapping...")
    encoder = load_encoder(checkpoint_path)

    image_paths = sorted(Path(camera_dir).glob("*.jpg"))[:max_images]
    all_results = []

    for i, img_path in enumerate(image_paths):
        print(f"\n[INFO] Occlusion {i+1}/{len(image_paths)}: {img_path.name}")

        result = compute_occlusion_sensitivity(
            encoder, str(img_path), device,
            occlusion_size=occlusion_size,
            stride=stride,
        )
        all_results.append(result)

        # Vẽ kết quả cho ảnh này
        output_path = f"{occ_dir}/occlusion_{img_path.stem}.png"
        create_occlusion_visualization(
            str(img_path), result, output_path,
            title=f"Occlusion Sensitivity — {img_path.name}",
        )
        print(f"    Max sensitivity: {result['max_sensitivity_val']:.2f} "
              f"at pixel ({result['max_sensitivity_pos'][0]}, {result['max_sensitivity_pos'][1]})")
        print(f"    Min sensitivity: {result['min_sensitivity_val']:.2f} "
              f"at pixel ({result['min_sensitivity_pos'][0]}, {result['min_sensitivity_pos'][1]})")

    # Vẽ biểu đồ tổng hợp
    print("\n[INFO] Creating summary chart...")
    create_sensitivity_summary(all_results, f"{occ_dir}/occlusion_summary.png")

    print(f"\n[DONE] Occlusion results saved to {occ_dir}/")


# ======================== CHẠY ========================
if __name__ == "__main__":
    ENCODER_CKPT = f"{WORKSPACE}/.cache/checkpoints/vjepa2/vitl_merge_3dataset_e50.pt"
    CAMERA_DIR = f"{WORKSPACE}/dataset_vf/data/CAMERA/CAM_P_F"

    run_occlusion_pipeline(
        checkpoint_path=ENCODER_CKPT,
        camera_dir=CAMERA_DIR,
        output_dir=OUTPUT_DIR,
        max_images=5,          # Giới hạn 5 ảnh vì occlusion mapping tốn GPU
        occlusion_size=32,     # 32px = 2 patches (cân bằng giữa chi tiết và tốc độ)
        stride=16,             # 16px = 1 patch (bước nhảy = patch_size)
    )
```

---

## 9. Script 7: Background Leakage Score (BLS)

> **Ý tưởng (từ Slide):**
> Sử dụng Cosine Similarity để đo mức độ tương đồng giữa các patch thuộc
> object (xe cộ, người) và các patch thuộc background (hạ tầng, chợ tự phát).
>
> **Công thức:**
> `BLS = (1/M) Σ CosineSimilarity(oᵢ, bᵢ)`
>
> Nếu BLS cao tại các vùng ranh giới đường mơ hồ hoặc chợ tự phát,
> chứng minh mô hình đang bị nhiễu thông tin nền nghiêm trọng.
>
> **Chiến thuật phân tách Object vs Background:**
> - Sử dụng DeepLabV3 (pretrained trên COCO, có sẵn trong `torchvision`)
>   để tạo segmentation mask tự động.
> - Mapping class labels → Object (person, car, truck, bus, motorcycle, bicycle)
>   vs Background (mọi thứ còn lại).
> - Chiếu mask pixel-level xuống patch-level (16×16 pixels/patch) bằng majority vote.

Paste vào **1 ô code** trên Colab:

```python
#!/usr/bin/env python3
"""
Script 7: Background Leakage Score (BLS).
Phân tách token object / background bằng DeepLabV3, tính BLS = Cosine(obj, bg).

Input:  outputs/spatial_eval/vf_token_features.pt (từ Script 1)
Output: outputs/spatial_eval/bls/*.png + bls_results.pt
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision import transforms
import cv2

WORKSPACE = "/content/vinfast"
OUTPUT_DIR = f"{WORKSPACE}/outputs/spatial_eval"


# ========== COCO class IDs cho DeepLabV3 ==========
# DeepLabV3 (pretrained COCO) có 21 classes (including background=0)
# Ta phân loại thành OBJECT (phương tiện + người) vs BACKGROUND (hạ tầng + mọi thứ khác)
OBJECT_CLASSES = {
    7: "car",        # ô tô
    14: "motorbike", # xe máy (motorcycle)
    15: "person",    # người
    2: "bicycle",    # xe đạp
    6: "bus",        # xe buýt
    # COCO-format for DeepLabV3: truck is not a separate class in VOC,
    # but car (7) covers most vehicles
}
# Tất cả class khác (0=background, 1=aeroplane, 3=bird, ..., 20=tvmonitor) → BACKGROUND


def load_segmentation_model(device: str = "cuda"):
    """
    Load DeepLabV3-ResNet101 pretrained trên COCO (có sẵn trong torchvision).
    Không cần tải thêm gì — torchvision tự cache checkpoint.
    """
    from torchvision.models.segmentation import deeplabv3_resnet101
    model = deeplabv3_resnet101(pretrained=True).to(device)
    model.eval()
    return model


def segment_image(model, image_path: str, device: str = "cuda") -> np.ndarray:
    """
    Chạy semantic segmentation trên 1 ảnh.
    Returns:
        seg_map: [H, W] — mỗi pixel = class ID (0-20)
    """
    img = Image.open(image_path).convert("RGB")
    W_orig, H_orig = img.size

    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)["out"]  # [1, 21, H, W]
        seg_map = output.argmax(dim=1).squeeze(0).cpu().numpy()  # [H, W]

    return seg_map  # [256, 512]


def create_patch_level_mask(
    seg_map: np.ndarray,
    patch_size: int = 16,
) -> np.ndarray:
    """
    Chuyển segmentation mask pixel-level → patch-level bằng majority vote.
    Mỗi patch 16×16 được gán label "object" nếu > 30% pixels thuộc object class.

    Returns:
        patch_mask: [H_patches, W_patches] — True nếu là object patch
    """
    H, W = seg_map.shape
    H_patches = H // patch_size
    W_patches = W // patch_size

    # Tạo binary mask: 1 = object, 0 = background
    object_mask = np.zeros_like(seg_map, dtype=np.float32)
    for cls_id in OBJECT_CLASSES:
        object_mask[seg_map == cls_id] = 1.0

    # Majority vote cho mỗi patch
    patch_mask = np.zeros((H_patches, W_patches), dtype=bool)
    for r in range(H_patches):
        for c in range(W_patches):
            patch_region = object_mask[
                r * patch_size : (r + 1) * patch_size,
                c * patch_size : (c + 1) * patch_size,
            ]
            # Ngưỡng 30%: nếu > 30% pixel trong patch là object → object patch
            patch_mask[r, c] = patch_region.mean() > 0.3

    return patch_mask


def compute_bls(
    features: torch.Tensor,
    patch_mask: np.ndarray,
    H_patches: int,
    W_patches: int,
) -> dict:
    """
    Tính Background Leakage Score (BLS).

    BLS = (1/M) Σ CosineSimilarity(oᵢ, bⱼ)
    với: oᵢ ∈ S_obj (token object), bⱼ ∈ S_bg (token background)

    BLS cao → ranh giới obj/bg bị mờ → mô hình bị nhiễu nền.
    BLS thấp → obj/bg rõ ràng → mô hình phân biệt tốt.

    Args:
        features:   [N, D] — token features 1 ảnh
        patch_mask: [H_patches, W_patches] — True = object
        H_patches, W_patches: kích thước lưới
    Returns:
        dict với BLS score, chi tiết
    """
    # Flatten mask
    mask_flat = patch_mask.reshape(-1)  # [N]
    N = min(features.shape[0], len(mask_flat))
    features = features[:N]
    mask_flat = mask_flat[:N]

    # Tách object / background tokens
    obj_indices = np.where(mask_flat)[0]
    bg_indices = np.where(~mask_flat)[0]

    num_obj = len(obj_indices)
    num_bg = len(bg_indices)

    if num_obj == 0 or num_bg == 0:
        return {
            "bls": 0.0,
            "num_obj_tokens": num_obj,
            "num_bg_tokens": num_bg,
            "obj_bg_cos_matrix": None,
            "warning": "Không đủ object hoặc background tokens",
        }

    # Trích token object và background
    obj_features = features[obj_indices]  # [num_obj, D]
    bg_features = features[bg_indices]    # [num_bg, D]

    # Chuẩn hóa L2
    obj_norm = F.normalize(obj_features, p=2, dim=-1)
    bg_norm = F.normalize(bg_features, p=2, dim=-1)

    # Tính cosine similarity giữa tất cả cặp (oᵢ, bⱼ)
    cos_matrix = obj_norm @ bg_norm.T  # [num_obj, num_bg]

    # BLS = trung bình toàn bộ cặp
    bls = cos_matrix.mean().item()

    # Phân tích thêm
    # BLS cao nhất ở cặp nào? (tìm object token bị "rò rỉ" nhiều nhất)
    max_cos_per_obj = cos_matrix.max(dim=1).values  # [num_obj]
    leaky_obj_idx = max_cos_per_obj.argmax().item()

    return {
        "bls": bls,
        "num_obj_tokens": num_obj,
        "num_bg_tokens": num_bg,
        "obj_bg_cos_matrix": cos_matrix,
        "max_cos_per_obj": max_cos_per_obj,
        "leaky_obj_global_idx": int(obj_indices[leaky_obj_idx]),
        "leaky_obj_max_cos": max_cos_per_obj[leaky_obj_idx].item(),
    }


def create_bls_visualization(
    image_path: str,
    seg_map: np.ndarray,
    patch_mask: np.ndarray,
    bls_result: dict,
    H_patches: int,
    W_patches: int,
    output_path: str,
    title: str = "",
):
    """Vẽ kết quả BLS: segmentation mask, patch-level mask, BLS heatmap."""

    img = np.array(Image.open(image_path).convert("RGB"))
    img_resized = cv2.resize(img, (512, 256))
    H, W = 256, 512

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. Ảnh gốc
    axes[0, 0].imshow(img_resized)
    axes[0, 0].set_title("Ảnh gốc (Camera VinFast)", fontsize=12)
    axes[0, 0].axis("off")

    # 2. Segmentation mask (từ DeepLabV3)
    seg_colored = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    for cls_id in OBJECT_CLASSES:
        seg_colored[seg_map == cls_id] = [255, 0, 0]  # Đỏ = object
    seg_colored[seg_map == 0] = [50, 50, 50]           # Xám đen = background
    # Các class khác (sidewalk, building, sky...) giữ xám nhạt
    for cls_id in range(21):
        if cls_id not in OBJECT_CLASSES and cls_id != 0:
            seg_colored[seg_map == cls_id] = [100, 150, 100]

    seg_overlay = cv2.addWeighted(img_resized, 0.6, seg_colored, 0.4, 0)
    axes[0, 1].imshow(seg_overlay)
    axes[0, 1].set_title("Semantic Segmentation\n(ĐỎ = Object, XÁM = Background)", fontsize=12)
    axes[0, 1].axis("off")

    # 3. Patch-level mask
    patch_vis = np.zeros((H_patches, W_patches, 3), dtype=np.uint8)
    patch_vis[patch_mask] = [255, 80, 80]     # Đỏ = object patch
    patch_vis[~patch_mask] = [80, 80, 200]    # Xanh = background patch
    patch_resized = cv2.resize(patch_vis, (W, H), interpolation=cv2.INTER_NEAREST)
    patch_overlay = cv2.addWeighted(img_resized, 0.5, patch_resized, 0.5, 0)
    axes[1, 0].imshow(patch_overlay)
    axes[1, 0].set_title(f"Patch-Level Mask\n"
                         f"Object: {bls_result['num_obj_tokens']} patches, "
                         f"Background: {bls_result['num_bg_tokens']} patches",
                         fontsize=12)
    axes[1, 0].axis("off")

    # 4. BLS heatmap (cosine leakage per-object-patch)
    if bls_result.get("max_cos_per_obj") is not None and bls_result["num_obj_tokens"] > 0:
        # Tạo leakage map: chỉ object patches mới có giá trị
        leakage_map = np.zeros(H_patches * W_patches, dtype=np.float32)
        obj_indices = np.where(patch_mask.reshape(-1))[0]
        max_cos = bls_result["max_cos_per_obj"].numpy()
        for idx_in_obj, global_idx in enumerate(obj_indices):
            if idx_in_obj < len(max_cos):
                leakage_map[global_idx] = max_cos[idx_in_obj]

        leakage_2d = leakage_map.reshape(H_patches, W_patches)
        leakage_resized = cv2.resize(leakage_2d, (W, H), interpolation=cv2.INTER_CUBIC)

        im = axes[1, 1].imshow(leakage_resized, cmap="YlOrRd", vmin=0, vmax=1)
        plt.colorbar(im, ax=axes[1, 1], label="Max Cosine(obj, bg)")
        axes[1, 1].set_title(f"Background Leakage Heatmap\n"
                             f"BLS = {bls_result['bls']:.4f} "
                             f"(cao = rò rỉ nhiều)",
                             fontsize=12)
    else:
        axes[1, 1].text(0.5, 0.5, "Không đủ dữ liệu\n(thiếu object/background)",
                        ha="center", va="center", fontsize=14, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Background Leakage Heatmap")
    axes[1, 1].axis("off")

    plt.suptitle(title or "Background Leakage Score (BLS) Analysis",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_bls_summary(
    all_bls_results: list,
    filenames: list,
    output_path: str,
):
    """Vẽ biểu đồ tổng hợp BLS cho tất cả ảnh."""

    bls_scores = [r["bls"] for r in all_bls_results]
    obj_counts = [r["num_obj_tokens"] for r in all_bls_results]
    bg_counts = [r["num_bg_tokens"] for r in all_bls_results]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # 1. BLS bar chart
    colors = ["red" if s > 0.5 else "orange" if s > 0.3 else "green" for s in bls_scores]
    x = range(len(bls_scores))
    axes[0].bar(x, bls_scores, color=colors)
    axes[0].axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Rò rỉ nghiêm trọng (0.5)")
    axes[0].axhline(y=0.3, color="orange", linestyle="--", alpha=0.7, label="Rò rỉ trung bình (0.3)")
    axes[0].set_xlabel("Image Index")
    axes[0].set_ylabel("Background Leakage Score (BLS)")
    axes[0].set_title("BLS Per Image\n(Cao = ranh giới obj/bg mờ → mô hình bị nhiễu nền)")
    axes[0].legend(fontsize=9)

    # 2. Object / Background token ratio
    axes[1].bar(x, obj_counts, label="Object tokens", color="coral", alpha=0.8)
    axes[1].bar(x, bg_counts, bottom=obj_counts, label="Background tokens", color="steelblue", alpha=0.8)
    axes[1].set_xlabel("Image Index")
    axes[1].set_ylabel("Number of Tokens")
    axes[1].set_title("Object vs Background Token Count\n(Tỷ lệ token thuộc đối tượng / nền)")
    axes[1].legend()

    plt.suptitle("Tổng Hợp Background Leakage Score — Toàn Bộ Dataset",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_bls_pipeline(
    features_path: str,
    camera_dir: str,
    output_dir: str,
    max_images: int = 10,
):
    """Pipeline chính: segment → tách obj/bg → tính BLS → vẽ."""

    bls_dir = f"{output_dir}/bls"
    os.makedirs(bls_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load features
    data = torch.load(features_path, map_location="cpu")
    features = data["features"]     # [num_images, N, D]
    filenames = data["filenames"]
    num_images, N, D = features.shape
    H_patches, W_patches = 16, 32
    print(f"[INFO] Loaded {num_images} images, {N} tokens/image")

    # 2. Load segmentation model
    print("[INFO] Loading DeepLabV3-ResNet101 (pretrained COCO)...")
    seg_model = load_segmentation_model(device)
    print("[INFO] Segmentation model loaded!")

    # 3. Xử lý từng ảnh
    image_paths = sorted(Path(camera_dir).glob("*.jpg"))
    img_map = {p.name: str(p) for p in image_paths}

    all_bls_results = []
    n_to_process = min(max_images, num_images)

    # Xuất bảng header
    print("\n" + "=" * 95)
    print(f"{'Filename':<35} {'BLS':>8} {'Obj Tokens':>12} {'BG Tokens':>12} {'Leaky?':>10}")
    print("=" * 95)

    for i in range(n_to_process):
        fname = filenames[i]
        if fname not in img_map:
            continue

        img_path = img_map[fname]

        # a. Segment ảnh
        seg_map = segment_image(seg_model, img_path, device)

        # b. Chuyển sang patch-level mask
        patch_mask = create_patch_level_mask(seg_map, patch_size=16)

        # c. Tính BLS
        bls_result = compute_bls(features[i], patch_mask, H_patches, W_patches)
        bls_result["filename"] = fname
        all_bls_results.append(bls_result)

        # d. In kết quả
        flag = "⚠️ HIGH" if bls_result["bls"] > 0.5 else ""
        print(f"{fname:<35} {bls_result['bls']:>8.4f} "
              f"{bls_result['num_obj_tokens']:>12} "
              f"{bls_result['num_bg_tokens']:>12} "
              f"{flag:>10}")

        # e. Vẽ kết quả
        output_path = f"{bls_dir}/bls_{Path(fname).stem}.png"
        create_bls_visualization(
            img_path, seg_map, patch_mask, bls_result,
            H_patches, W_patches, output_path,
            title=f"BLS Analysis — {fname}",
        )

    print("=" * 95)

    # 4. Tổng hợp
    avg_bls = np.mean([r["bls"] for r in all_bls_results])
    print(f"\n[INFO] Average BLS across {len(all_bls_results)} images: {avg_bls:.4f}")
    if avg_bls > 0.5:
        print("[⚠️ WARNING] BLS trung bình > 0.5: Mô hình bị rò rỉ nền nghiêm trọng!")
        print("           Ranh giới object/background bị mờ → cần fine-tune trên dữ liệu VN.")
    elif avg_bls > 0.3:
        print("[⚠️ NOTICE] BLS trung bình 0.3-0.5: Mức rò rỉ trung bình, cần cải thiện.")
    else:
        print("[✅ OK] BLS trung bình < 0.3: Mô hình phân biệt obj/bg khá tốt.")

    # 5. Vẽ biểu đồ tổng hợp
    create_bls_summary(
        all_bls_results,
        [r["filename"] for r in all_bls_results],
        f"{bls_dir}/bls_summary.png",
    )

    # 6. Lưu kết quả
    save_results = []
    for r in all_bls_results:
        save_r = {k: v for k, v in r.items() if k != "obj_bg_cos_matrix"}
        if "max_cos_per_obj" in save_r and save_r["max_cos_per_obj"] is not None:
            save_r["max_cos_per_obj"] = save_r["max_cos_per_obj"].numpy().tolist()
        save_results.append(save_r)
    torch.save(save_results, f"{bls_dir}/bls_results.pt")

    print(f"\n[DONE] BLS results saved to {bls_dir}/")


# ======================== CHẠY ========================
if __name__ == "__main__":
    FEATURES_PATH = f"{OUTPUT_DIR}/vf_token_features.pt"
    CAMERA_DIR = f"{WORKSPACE}/dataset_vf/data/CAMERA/CAM_P_F"

    run_bls_pipeline(
        features_path=FEATURES_PATH,
        camera_dir=CAMERA_DIR,
        output_dir=OUTPUT_DIR,
        max_images=10,
    )
```

---

## 10. Script Tổng Hợp: Chạy Toàn Bộ Pipeline
Tạo file `run_all_spatial_eval.sh` và chạy bằng lệnh `bash run_all_spatial_eval.sh`:

```bash
#!/bin/bash
# run_all_spatial_eval.sh

echo "========================================="
echo "   V-JEPA2 COMPLETE SPATIAL EVALUATION   "
echo "========================================="

echo ""
echo "===== STEP 1/7: Extract Token Features ====="
python spatial_eval/extract_features.py

echo ""
echo "===== STEP 2/7: Cosine Similarity ====="
python spatial_eval/cosine_analysis.py

echo ""
echo "===== STEP 3/7: Mahalanobis Distance ====="
python spatial_eval/mahalanobis_analysis.py

echo ""
echo "===== STEP 4/7: Attention & OOD Heatmaps ====="
python spatial_eval/attention_heatmaps.py

echo ""
echo "===== STEP 5/7: PCA Visualization ====="
python spatial_eval/pca_analysis.py

echo ""
echo "===== STEP 6/7: Occlusion Mapping ====="
python spatial_eval/occlusion_mapping.py

echo ""
echo "===== STEP 7/7: Background Leakage Score ====="
python spatial_eval/bls_analysis.py

echo ""
echo "===== HOÀN TẤT ====="
echo "Kết quả tại: /content/vinfast/outputs/spatial_eval/"
```

## 11. Bảng Tổng Hợp Sản Phẩm Đầu Ra (Deliverables)
Tất cả các file sẽ được lưu tại thư mục `outputs/spatial_eval/`.

| File / Folder | Kỹ Thuật | Ý Nghĩa / Mục Đích (Dành cho Slide báo cáo) |
|---|---|---|
| `vf_token_features.pt` | Core | Ma trận token trích xuất [50, 512, 1024]. |
| `inter_image_cosine_matrix.png` | Cosine | Đánh giá tính đồng nhất của tập dữ liệu. |
| `intra_image_cosine_sample.png` | Cosine | OOD detection dựa trên Cosine distance patch-level. |
| `ood_scores_bar.png` | OOD Metrics | Biểu đồ tỷ lệ patch vượt ngưỡng OOD per-image. |
| `mahalanobis_scores_bar.png` | Mahalanobis | Biểu đồ điểm Mahalanobis per-image. |
| `heatmaps/attention_*.png` | Attention | Cho thấy sự phân bổ chú ý hỗn loạn vào cảnh vật biên. |
| `heatmaps/ood_*.png` | Anomaly Map | Nhấn mạnh vật thể phi tiêu chuẩn (xe ba gác, chở cồng kềnh). |
| `pca/pca_*.png` | PCA | Trực quan 3D RGB đặc trưng. Vùng nhiễu màu loang lổ = dị biệt. |
| `pca_global_scatter.png` | PCA | Cho thấy sự trôi dạt của khung hình lộn xộn khỏi cụm trung tâm. |
| `occlusion/occlusion_*.png` | Occlusion | Sensitivity map chứng minh bias khu vực trung tâm. |
| `occlusion_summary.png` | Occlusion | Biểu đồ độ nhạy theo cột, phớt lờ rủi ro mép biên ảnh. |
| `bls/bls_*.png` | BLS | Ảnh phân tách đối tượng và nền (DeepLabV3). |
| `bls_summary.png` | BLS | Điểm rò rỉ nền ($L_{leak} < 0.3$, chứng tỏ tách nền tốt). |
