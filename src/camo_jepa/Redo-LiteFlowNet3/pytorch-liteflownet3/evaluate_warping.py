import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
from tqdm import tqdm

def load_image(img_path):
    """Load image as PyTorch tensor [1, 3, H, W] normalized to [0, 1]"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    # to tensor [1, C, H, W]
    ten_img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return ten_img

def warp_image(img, flow):
    """
    Dùng hàm grid_sample của PyTorch để warp (kéo ngược) img theo flow.
    img: Tensor [B, C, H, W] (ở đây B=1)
    flow: Tensor [B, 2, H, W] chứa vector di chuyển (pixel)
    """
    B, C, H, W = img.size()
    
    # Tạo tọa độ lưới ban đầu [0, W-1] và [0, H-1]
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(img.device)
    
    # Cộng thêm vector Optical Flow
    vgrid = grid + flow
    
    # Chuẩn hóa tọa độ về khoảng [-1, 1] cho hàm grid_sample
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    
    # Đổi shape sang [B, H, W, 2]
    vgrid = vgrid.permute(0, 2, 3, 1)
    
    # Thực hiện Warp
    warped_img = F.grid_sample(img, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    # Tạo Mask (Bỏ qua các pixel bị văng ra khỏi khung hình sau khi warp)
    mask = (vgrid[..., 0] >= -1) & (vgrid[..., 0] <= 1) & (vgrid[..., 1] >= -1) & (vgrid[..., 1] <= 1)
    mask = mask.unsqueeze(1).float()
    
    return warped_img, mask

def compute_photometric_error(img1, warped_img2, mask):
    """Tính L1 Loss giữa Ảnh 1 và Ảnh 2-Warped, bỏ qua các điểm mù"""
    diff = torch.abs(img1 - warped_img2) * mask
    # Tính trung bình dựa trên tổng số điểm hợp lệ
    l1_error = diff.sum() / (mask.sum() * 3 + 1e-8)  # *3 vì có 3 kênh RGB
    return l1_error.item()

def main():
    parser = argparse.ArgumentParser(description="Tính Photometric Warping Error cho dữ liệu không có Ground Truth")
    parser.add_argument('--img_dir', type=str, required=True, help="Thư mục chứa ảnh gốc (Frame 1, Frame 2...)")
    parser.add_argument('--flow_dir', type=str, required=True, help="Thư mục chứa các file .npy đã tính")
    parser.add_argument('--save_warp_dir', type=str, default=None, help="(Optional) Thư mục để lưu ảnh đã Warp nhằm kiểm chứng bằng mắt")
    args = parser.parse_args()

    # Quét ảnh đầu vào (hỗ trợ png, jpg, jpeg)
    extensions = ('*.png', '*.jpg', '*.jpeg')
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(args.img_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.img_dir, ext.upper())))
    image_files = sorted(image_files)

    if len(image_files) < 2:
        print("Cần ít nhất 2 ảnh trong thư mục gốc để đánh giá.")
        return
        
    if args.save_warp_dir:
        os.makedirs(args.save_warp_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang tính Photometric Error bằng thiết bị: {device}")

    total_error = 0.0
    valid_count = 0

    for i in tqdm(range(len(image_files) - 1)):
        img1_path = image_files[i]
        img2_path = image_files[i+1]
        
        base_name = os.path.basename(img1_path)
        name_no_ext = os.path.splitext(base_name)[0]
        flow_path = os.path.join(args.flow_dir, f"flow_{name_no_ext}.npy")

        if not os.path.exists(flow_path):
            continue

        # Load tensors
        img1 = load_image(img1_path).to(device)
        img2 = load_image(img2_path).to(device)
        flow_np = np.load(flow_path)
        flow = torch.from_numpy(flow_np).unsqueeze(0).to(device)  # [1, 2, H, W]

        # Warp Frame 2 ngược về Frame 1
        warped_img2, mask = warp_image(img2, flow)

        # Tính L1 Loss
        error = compute_photometric_error(img1, warped_img2, mask)
        total_error += error
        valid_count += 1
        
        # Lưu ảnh Warped nếu có cờ `--save_warp_dir`
        if args.save_warp_dir:
            # warped_img2 có dạng [1, 3, H, W], dải màu [0, 1]. Cần chuyển về [H, W, 3] dải [0, 255] hệ BGR
            out_img = warped_img2[0].permute(1, 2, 0).cpu().numpy() * 255.0
            out_img = cv2.cvtColor(out_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.save_warp_dir, f"warped_{name_no_ext}.png"), out_img)

    if valid_count > 0:
        avg_error = total_error / valid_count
        print(f"\n==================================================")
        print(f"Photometric Warping Error (L1 Loss trung bình): {avg_error:.5f}")
        print(f"Sai số pixel trung bình (thang 0-255): {avg_error * 255:.2f} pixels / kênh màu")
        print(f"Đã chấm điểm cho {valid_count} frame.")
        print(f"==================================================")
    else:
        print("Không tìm thấy file .npy nào khớp với ảnh đầu vào.")

if __name__ == '__main__':
    main()
