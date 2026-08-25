import os
import numpy as np
import cv2
import argparse
import glob
from tqdm import tqdm

def make_color_wheel():
    """Tạo bánh xe 55 màu chuẩn Middlebury của đại học Brown"""
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0, YG, 1)/YG)
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(0, CB, 1)/CB)
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
    col += BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(0, MR, 1)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def compute_color(u, v):
    """Tính toán mảng màu dựa trên thuật toán Middlebury"""
    colorwheel = make_color_wheel()
    nanFlow = np.isnan(u) | np.isnan(v)
    u[nanFlow] = 0
    v[nanFlow] = 0
    
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2 + v**2)
    a = np.arctan2(-v, -u) / np.pi
    
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    
    img = np.empty([np.size(u, 0), np.size(u, 1), 3])
    
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1 - f) * col0 + f * col1
        
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75
        img[:, :, i] = np.floor(255 * col).astype(np.uint8)
        
    return img

def flow_to_color(flow):
    """
    Chuyển đổi flow tensor (2, H, W) sang ảnh màu bằng chuẩn Middlebury.
    """
    u = flow[0, :, :]
    v = flow[1, :, :]
    
    # Chuẩn hóa để tránh bị vỡ/quá tối do nhiễu (Lọc 1% pixel siêu dị)
    rad = np.sqrt(u**2 + v**2)
    maxrad = np.percentile(rad, 99)
    if maxrad > 0:
        u = u / maxrad
        v = v / maxrad
        
    img = compute_color(u, v)
    
    # Đổi RGB (của Middlebury) sang BGR (để OpenCV lưu file không bị lộn màu đỏ/xanh)
    img_bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return img_bgr

def main():
    parser = argparse.ArgumentParser(description="Đánh giá định tính Optical Flow (Chỉ Heatmap)")
    parser.add_argument('--input_dir', type=str, required=True, help="Thư mục chứa các file .npy")
    parser.add_argument('--output_dir', type=str, required=True, help="Thư mục lưu ảnh màu Heatmap")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    npy_files = glob.glob(os.path.join(args.input_dir, '*.npy'))
    if not npy_files:
        print(f"Không tìm thấy file .npy nào trong {args.input_dir}")
        return

    print(f"Đang tiến hành vẽ Heatmap cho {len(npy_files)} file Optical Flow...")
    
    for npy_file in tqdm(npy_files):
        flow = np.load(npy_file)
        
        if flow.ndim != 3 or flow.shape[0] != 2:
            continue
            
        color_img = flow_to_color(flow)
        
        base_name = os.path.basename(npy_file).replace('.npy', '')
        cv2.imwrite(os.path.join(args.output_dir, f"{base_name}.png"), color_img)

if __name__ == '__main__':
    main()
