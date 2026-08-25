import os
import glob
import numpy as np
import torch
import PIL.Image
import math
import sys
from tqdm import tqdm

from run import Network

def load_image(img_path):
    # Đọc ảnh và chuyển sang định dạng tensor BGR [0, 1] như yêu cầu của LiteFlowNet3
    img = np.array(PIL.Image.open(img_path))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    return torch.FloatTensor(np.ascontiguousarray(img))

def estimate_flow(netNetwork, tenFirst, tenSecond):
    intWidth = tenFirst.shape[2]
    intHeight = tenFirst.shape[1]

    tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

    # Pad ảnh lên bội số của 32
    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

    tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    # Chạy inference
    tenFlow = torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)
    
    # Rescale lại flow vectors theo tỷ lệ padding
    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[0, :, :, :].cpu().numpy()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Trích xuất Optical Flow bằng LiteFlowNet3")
    parser.add_argument('--img_dir', type=str, required=True, help="Thư mục chứa ảnh gốc (ví dụ: data/vf/images/train/vf_recording_000/)")
    parser.add_argument('--out_dir', type=str, required=True, help="Thư mục lưu file .npy kết quả")
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    WEIGHTS_PATH = os.path.join(SCRIPT_DIR, 'network-sintel.pytorch')
    DATA_DIR = args.img_dir
    OUTPUT_DIR = args.out_dir
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Khởi tạo model
    if not os.path.isfile(WEIGHTS_PATH):
        print(f"ERROR: Không tìm thấy file weights tại {WEIGHTS_PATH}")
        print(f"Hãy tải file về bằng lệnh: gdown 1vUSEIxXGZa9d2PQ82SG_gbbIUWLNfH50")
        return
    
    print(f"Loading LiteFlowNet3 model from {WEIGHTS_PATH}...")
    netNetwork = Network().cuda().eval()
    netNetwork.load_state_dict(torch.load(WEIGHTS_PATH))
    print("Model loaded successfully!")

    # Lấy danh sách ảnh (hỗ trợ png, jpg, jpeg)
    extensions = ('*.png', '*.jpg', '*.jpeg')
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(DATA_DIR, ext)))
        # Bắt thêm trường hợp in hoa .PNG, .JPG
        image_files.extend(glob.glob(os.path.join(DATA_DIR, ext.upper())))
    
    # Sắp xếp theo tên file để đảm bảo đúng thứ tự thời gian
    image_files = sorted(image_files)
    
    if len(image_files) < 2:
        print(f"Không tìm thấy đủ ảnh trong {DATA_DIR} để tính optical flow!")
        return
        
    print(f"Found {len(image_files)} images. Processing {len(image_files)-1} pairs...")
    
    for i in tqdm(range(len(image_files) - 1)):
        img1_path = image_files[i]
        img2_path = image_files[i+1]
        
        # Load tensors
        tenFirst = load_image(img1_path)
        tenSecond = load_image(img2_path)
        
        # Calculate flow
        with torch.no_grad():
            flow_matrix = estimate_flow(netNetwork, tenFirst, tenSecond)
            
        # Lưu file .npy (shape: [2, H, W]) — dùng splitext để hỗ trợ .jpg, .png
        base_name = os.path.splitext(os.path.basename(img1_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"flow_{base_name}.npy")
        np.save(out_path, flow_matrix)

if __name__ == '__main__':
    main()
