# Phân tích kiến trúc `pytorch-liteflownet3`

> Phân tích file `src/pytorch-liteflownet3/run.py` — Cập nhật 21/08/2026

---

## 1. Câu trả lời nhanh: Có nên lập môi trường ảo không?

**CÓ, bắt buộc phải lập môi trường ảo trước khi compile!**

Lý do chính:
- `correlation_package` được compile bằng `python setup.py install` — lệnh này sẽ cài thẳng vào môi trường Python đang active.
- Nếu không có venv, package sẽ cài vào Python hệ thống. Nếu sau này muốn xóa hay đổi phiên bản PyTorch/CUDA, bạn sẽ phải compile lại.
- Môi trường ảo đảm bảo version của PyTorch, CUDA, và `correlation_package` luôn khớp nhau.

```bash
# Tạo venv (chạy 1 lần)
cd vf-world-models
python -m venv venv_camo
source venv_camo/bin/activate

# Cài PyTorch phù hợp với CUDA của máy
pip install torch torchvision

# Sau đó mới compile correlation layer
cd src/pytorch-liteflownet3/correlation_package
python setup.py install
```

---

## 2. Tổng quan toàn bộ pipeline (Full Forward Pass)

```
Frame 1 [C, H, W]  ──┐
                       ├──→ Feature Pyramid (6 cấp độ) ──→ [F1_1...F1_6]
Frame 2 [C, H, W]  ──┘                                     [F2_1...F2_6]
                                                               │
                             ┌─────────────────────────────────┘
                             ▼  (4 vòng lặp, từ thô → mịn: level 6→3)
                    ┌──────────────┐    ┌──────────────┐    ┌────────────────┐
                    │  Matching    │───→│  Subpixel    │───→│ Regularization │
                    │ (CM module)  │    │ refinement   │    │ (FD module)    │
                    └──────────────┘    └──────────────┘    └────────────────┘
                             │
                             ▼
                    Flow [2, H, W] × 20.0  (u, v pixel displacement)
```

---

## 3. Phân tích chi tiết từng module

### Module 1: `Features` — Kim tự tháp đặc trưng

**Mục đích:** Trích xuất đặc trưng từ mỗi ảnh ở 6 độ phân giải khác nhau.

```
Input Image [3, H, W]
  → netOne: Conv2D(3→32, 7×7, stride=1)  + LeakyReLU  → [32, H, W]
  → netTwo: Conv2D(32→32, stride=2)      + LeakyReLU  → [32, H/2, W/2]
  → netThr: Conv2D(32→64, stride=2)      + LeakyReLU  → [64, H/4, W/4]
  → netFou: Conv2D(64→96, stride=2)      + LeakyReLU  → [96, H/8, W/8]
  → netFiv: Conv2D(96→128, stride=2)     + LeakyReLU  → [128, H/16, W/16]
  → netSix: Conv2D(128→192, stride=2)    + LeakyReLU  → [192, H/32, W/32]
```

**Tại sao cần 6 cấp?** Đây là kỹ thuật "Coarse-to-Fine" — bắt đầu ước tính flow ở độ phân giải thấp (dễ, ít nhiễu), sau đó tinh chỉnh dần lên độ phân giải cao hơn.

---

### Module 2: `Matching` — Cost Volume Modulation (CM) ⭐ Đóng góp chính

**Đây là đóng góp lớn nhất của LiteFlowNet3 so với LiteFlowNet2!**

#### Bước 2a: Cross-Correlation (Cost Volume cơ bản)
```python
crossCorr = Correlation(pad_size=4, kernel_size=1, max_displacement=4)
tenCorrelation = crossCorr(tenFeaturesFirst, tenFeaturesSecond)
# Output: [B, 81, H_l, W_l] — 81 = (4*2+1)^2 vị trí dịch chuyển có thể
```
- Với mỗi pixel ở Frame 1, tính độ tương đồng với 81 pixel lân cận trong Frame 2.
- Kết quả là 81 "phiếu bầu" — mỗi phiếu là xác suất pixel đã dịch chuyển tới vị trí đó.

#### Bước 2b: Auto-Correlation → Confidence Map (Cái mới của LiteFlowNet3!)
```python
autoCorr = Correlation(pad_size=6, kernel_size=1, max_displacement=6, stride2=2)
tenCorrelation_self = autoCorr(tenFeaturesFirst, tenFeaturesFirst)
# So sánh một pixel với CHÍNH CÁC PIXEL LÂN CẬN của nó trong cùng 1 frame
```
- Nếu các pixel xung quanh một điểm **trông y chang nhau** (ví dụ: bức tường trơn, bầu trời) → pixel đó có `ambiguity` cao → confidence thấp.
- Nếu một điểm **trông độc đáo, khác biệt** với xung quanh → confidence cao → tin tưởng flow tại điểm đó.

#### Bước 2c: Cost Volume Modulation
```python
corrScalar = self.corrScalar(corrfeat)   # [B, 81, H, W] — scale factor
corrOffset = self.corrOffset(corrfeat)   # [B, 81, H, W] — offset
tenCorrelation = corrScalar * tenCorrelation + corroffset
```
- Dùng Confidence Map để **scale và shift** Cost Volume — pixel có ambiguity cao thì bị "phạt" (nhân scale nhỏ, cộng offset).
- Đây là cơ chế "lọc nhiễu" của LiteFlowNet3.

---

### Module 3: `Subpixel` — Tinh chỉnh sub-pixel

**Mục đích:** Flow ước tính từ `Matching` chỉ chính xác đến pixel. Module này tinh chỉnh thêm ở mức sub-pixel.

```python
# Warp Frame2 bằng flow hiện tại, sau đó so sánh lại với Frame1
tenFeaturesSecond_warped = backwarp(tenFeaturesSecond, tenFlow * scale)
netMain(concat[Frame1_features, Frame2_warped_features, tenFlow])
```

---

### Module 4: `Regularization` — Flow Field Deformation (FD) ⭐ Đóng góp thứ 2

**Đây là đóng góp thứ hai của LiteFlowNet3 — làm sắc nét biên giới của vật thể chuyển động.**

```python
# Tính pixel difference sau khi đã warp Frame2 theo flow hiện tại
tenDifference = (Frame1 - backwarp(Frame2, tenFlow)).pow(2).sum().sqrt()

# Tính "local displacement field" — mỗi pixel tham khảo flow của hàng xóm
tenDist = netDist(mainfeat).pow(2).neg().exp()  # softmax-like weights

# Thay thế flow của pixel "xấu" bằng weighted average flow của láng giềng
tenScaleX = netScaleX(tenDist * unfold(tenFlow_x)) * divisor
tenScaleY = netScaleY(tenDist * unfold(tenFlow_y)) * divisor
```

**Trực quan:** Hãy tưởng tượng mỗi pixel sẽ "hỏi thăm" 25 pixel xung quanh nó "bạn đang chuyển động theo hướng nào?", rồi lấy trung bình có trọng số. Pixel nào có `difference` lớn (khó tin) sẽ bị phụ thuộc nhiều vào hàng xóm hơn.

---

### Module 5: `backwarp` — Hàm tiện ích quan trọng

Hàm này xuất hiện xuyên suốt toàn bộ pipeline:

```python
def backwarp(tenInput, tenFlow):
    # Tạo lưới tọa độ chuẩn hóa [-1, 1]
    # Dịch chuyển lưới theo flow vector
    # Dùng grid_sample để lấy pixel value tại vị trí mới
    return F.grid_sample(tenInput, grid + tenFlow, mode='bilinear')
```

**Ý nghĩa:** Nếu flow nói "pixel tại (x,y) đã di chuyển đến (x+dx, y+dy)", thì backwarp sẽ lấy giá trị Frame2 tại vị trí `(x+dx, y+dy)` để so sánh với Frame1 tại `(x,y)`.

---

## 4. Input / Output Interface

### Input (hàm `estimate`)
```python
tenFirst  = [3, H, W]  # Frame t,   normalize [0, 1], channel BGR
tenSecond = [3, H, W]  # Frame t+1, normalize [0, 1], channel BGR
```

**Lưu ý quan trọng:** H và W **phải là bội số của 32**! Hàm `estimate` tự động pad lên bội số 32 rồi crop về kích thước gốc.

### Output
```python
tenFlow = [2, H, W]  # Channel 0: u (horizontal displacement in pixels)
                     # Channel 1: v (vertical displacement in pixels)
```
Giá trị flow được nhân thêm `× 20.0` trước khi trả về (rescaling của tác giả).

---

## 5. Vấn đề kỹ thuật: `correlation_package`

Repo này dùng custom CUDA kernel từ NVIDIA-FlowNet2:
```
correlation_package/
├── correlation.py              ← Python wrapper
├── correlation_cuda.cc         ← C++ binding
├── correlation_cuda_kernel.cu  ← CUDA kernel (1000+ lines)
├── correlation_cuda_kernel.cuh ← CUDA header
└── setup.py                    ← Build script
```

**Tại sao cần CUDA custom kernel?** Phép tính `Correlation` là một phép tính đặc biệt — so sánh patch giữa 2 feature map — không có sẵn trong PyTorch. Nó phải được viết thủ công bằng CUDA để đủ nhanh.

**Yêu cầu:** CUDA phải được cài đặt sẵn và `nvcc` phải hoạt động. Kiểm tra bằng:
```bash
nvcc --version
```

---

## 6. Kết luận & Kế hoạch tích hợp vào CaMo-JEPA

Điểm kết nối với `motion/flow.py` trong CaMo-JEPA:

| CaMo-JEPA cần | pytorch-liteflownet3 cung cấp |
|---|---|
| `FrameDifferenceFlow` (placeholder cần thay) | `Network` class trong `run.py` |
| `gt_flow [B, T-1, 2, H, W]` | `estimate(frame1, frame2)` → `[2, H, W]` |
| Chạy offline, không cần gradient | `torch.set_grad_enabled(False)` đã có sẵn |

---

## 7. Những thay đổi đã thực hiện so với repo gốc

> Repo gốc: [sniklaus/pytorch-liteflownet3](https://github.com/sniklaus/pytorch-liteflownet3) (chỉ hỗ trợ PyTorch 1.3.0, CUDA 9.0)
> Môi trường thực tế: Python 3.10, PyTorch 2.x, CUDA 13.3, GPU NVIDIA L4

### 7.1. `correlation_package/setup.py` — Fix lỗi biên dịch

| Thay đổi | Dòng | Lý do |
|---|---|---|
| `c++11` → `c++17` trong `cxx_args` | ~8 | PyTorch 2.x dùng `std::optional` là cú pháp C++17, C++11 không hiểu |
| Xóa `compute_50`, `compute_52` | ~10–15 | CUDA 12+ đã khai tử hỗ trợ kiến trúc Maxwell (GTX 900) |
| Xóa `compute_60`, `compute_61` | ~10–15 | CUDA 13 khai tử tiếp kiến trúc Pascal (GTX 1080) |
| Giữ lại & thêm `compute_89` | ~10–15 | Đây là kiến trúc Ada Lovelace của GPU **NVIDIA L4** |

**File sau khi sửa (`nvcc_args`):**
```python
cxx_args = ['-std=c++17']

nvcc_args = [
    '-gencode', 'arch=compute_89,code=sm_89',
    '-gencode', 'arch=compute_89,code=compute_89'
]
```

### 7.2. `correlation_package/correlation_cuda_kernel.cu` — Fix API cũ

| Thay đổi | Lệnh sed đã dùng | Lý do |
|---|---|---|
| `.type()` → `.scalar_type()` | `sed -i 's/\.type()/.scalar_type()/g'` | PyTorch 2.x đã loại bỏ API `.type()` |

```bash
# Lệnh đã chạy trên server
sed -i 's/\.type()/.scalar_type()/g' correlation_cuda_kernel.cu
sed -i 's/\.type()/.scalar_type()/g' correlation_cuda.cc
```

> **Ghi chú:** Sau khi sửa vẫn còn nhiều `warning` về `.data()` → `.data_ptr()`, nhưng đây là warning vô hại, không ảnh hưởng đến kết quả.

### 7.3. Script mới được thêm vào repo: `extract_optical_flow.py`

File này **không có trong repo gốc**, được mình viết mới hoàn toàn nằm tại `src/pytorch-liteflownet3/extract_optical_flow.py`.

**Chức năng:** Nhận một thư mục chứa ảnh `.png` đã sắp xếp theo thứ tự thời gian, tự động tính Optical Flow giữa từng cặp ảnh liên tiếp (t và t+1), lưu kết quả ra file `.npy` với shape `[2, H, W]`.

```python
# Chạy lệnh này để trích xuất flow
python extract_optical_flow.py
# Output: /home/thanhpnc/src/data/optical_flow/flow_<frame_name>.npy
```

### 7.4. Môi trường ảo và cách kích hoạt lại

```bash
# Kích hoạt môi trường
conda activate camo_env

# BẮT BUỘC phải chạy lệnh này mỗi lần mở terminal mới trước khi dùng model
export LD_LIBRARY_PATH=/home/thanhpnc/miniconda/envs/camo_env/lib:$LD_LIBRARY_PATH

# Hoặc để tự động, thêm vào ~/.bashrc:
echo 'export LD_LIBRARY_PATH=/home/thanhpnc/miniconda/envs/camo_env/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

---

## 8. Các bước tiếp theo (Next Steps)

### Phase 1: Test End-to-End trên dữ liệu thật (Ưu tiên cao)

- [ ] **[Server]** Thêm dòng `export LD_LIBRARY_PATH=...` vào `~/.bashrc` để không phải gõ tay mỗi lần.
- [ ] **[Server]** Chuẩn bị ảnh thực tế (xem yêu cầu dữ liệu ở Mục 9 bên dưới), chạy `extract_optical_flow.py` để tạo ra tập `.npy`.
- [ ] **[Mac]** Sửa `src/camo_jepa/motion/flow.py`: Thay thế `FrameDifferenceFlow` bằng module đọc file `.npy` đã tính sẵn.
- [ ] **[Mac]** Sửa `src/camo_jepa/pipeline/phase1.py`: Cập nhật hàm `forward()` để nhận thêm `flow_tensors` từ Dataloader.
- [ ] **[Mac]** Viết `CaMoJEPADataset` — PyTorch Dataset class đọc cả ảnh lẫn file `.npy` flow tương ứng.
- [ ] **[Mac]** Viết training loop cơ bản (`train.py`), test với 1 batch nhỏ.

### Phase 2: Đánh giá & Chuẩn bị huấn luyện thật

- [ ] Chạy chay (Dry run) 1 epoch với batch size = 2, kiểm tra Loss có hội tụ không.
- [ ] Viết script đánh giá (Evaluation) — tính EPE (End-Point Error) trên tập validation.
- [ ] Push toàn bộ code lên Git repo, viết `requirements.txt` / `environment.yml`.

---

## 9. ⚠️ Yêu cầu về Dữ liệu — Ghi chú cho Chị Ngọc (Lead)

> [!IMPORTANT]
> **Đây là những yêu cầu kỹ thuật bắt buộc về định dạng dữ liệu đầu vào để pipeline CaMo-JEPA hoạt động đúng.**

### 9.1. Định dạng ảnh
- **Định dạng file:** `.png` hoặc `.jpg` (PNG khuyến khích vì không mất dữ liệu)
- **Màu sắc:** RGB, 3 channel (không dùng ảnh grayscale)
- **Độ phân giải:** Tối thiểu `256×256`. **Phải là bội số của 32** — LiteFlowNet3 yêu cầu cứng điều này. Các giá trị an toàn: `256×256`, `320×256`, `384×256`, `512×384`, v.v.

### 9.2. Cấu trúc thư mục & đặt tên file
```
data/
├── video_001/
│   ├── 00001.png   ← Frame 1
│   ├── 00002.png   ← Frame 2
│   ├── 00003.png   ← Frame 3
│   └── ...
├── video_002/
│   ├── 00001.png
│   └── ...
```
- **File ảnh PHẢI được đặt tên theo thứ tự số** để đảm bảo sắp xếp theo đúng thứ tự thời gian khi dùng `glob.glob` + `sorted()`.
- **Mỗi video/clip phải nằm trong một thư mục riêng** (không trộn lẫn các video với nhau trong cùng 1 thư mục).

### 9.3. Số lượng frame tối thiểu cho 1 sample
- CaMo-JEPA cần `context_frames + pred_frames` = hiện tại là `4 + 2 = 6 frames` cho mỗi lần huấn luyện.
- **Mỗi clip phải có ít nhất 7 frame liên tiếp** (6 để train, có buffer dư ra cho trường hợp flow bị lỗi ở frame đầu/cuối).

### 9.4. Yêu cầu về chất lượng video
- **Frame rate:** Nên ổn định (ví dụ: 10 FPS hoặc 30 FPS). Tránh video bị chụp không đều (variable frame rate) vì sẽ làm magnitude của Optical Flow không nhất quán giữa các sample.
- **Chuyển động:** Model hoạt động tốt nhất với video có chuyển động rõ ràng nhưng không quá nhanh (ví dụ: lái xe ở tốc độ thành phố). Cảnh tĩnh hoặc chuyển động camera cực nhanh (blur) sẽ làm độ chính xác của Optical Flow giảm.
- **Ánh sáng:** Tránh video có thay đổi ánh sáng đột ngột (ví dụ: đi từ trong tunnel ra nắng). LiteFlowNet3 dựa trên giả thuyết "brightness constancy" — ánh sáng của một điểm không đổi khi nó chuyển động.

### 9.5. Dung lượng ổ cứng cần chuẩn bị
- Mỗi file `.npy` flow của 1 cặp ảnh `256×256` ≈ **512 KB**.
- Nếu tập dữ liệu có **N ảnh**, sẽ sinh ra **N-1 file `.npy`**.
- Ví dụ: 10,000 ảnh → ~5 GB file `.npy` flow.
- **Khuyến nghị:** Cần ít nhất `[dung lượng ảnh gốc] × 1.5` dung lượng trống trên ổ cứng Server.

### 9.6. Dataset công khai có thể dùng để test
Nếu chưa có dữ liệu thực tế, có thể dùng các dataset benchmark có sẵn để chạy thử:
| Dataset | Loại | Link |
|---|---|---|
| **nuScenes** | Lái xe tự động | https://www.nuscenes.org |
| **CARLA** | Simulator lái xe | https://carla.org |
| **MPI Sintel** | Animation (đã dùng để train model) | http://sintel.is.tue.mpg.de |
| **KITTI 2015** | Lái xe, có ground truth flow | https://www.cvlibs.net/datasets/kitti |
