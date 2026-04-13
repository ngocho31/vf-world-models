# Phân chia công việc

## Công việc tuần 14/04/26 - 26/04/26

| Người phụ trách | Công việc |
| :--- | :--- |
| Phượng | [P2](#p2-thích-nghi-nhận-thức) |
| Nam | [P3](#p3-nghiên-cứu-hành-vi-giả-lập) |
| Thành | [P4](#p4-nghiên-cứu-môi-trường-đánh-giá) |
| Ân | [P5](#p5-nghiên-cứu-hợp-nhất) |
| Ngọc | [P1](#p1-hạ-tầng-và-tiền-xử-lý-dữ-liệu) |

---

## Chi tiết nhiệm vụ

### P1. Hạ tầng và Tiền xử lý dữ liệu

* **Nhiệm vụ:**
  * **Trích xuất & Đồng bộ:** Khớp Timestamp giữa Camera - LiDAR - CSV (Steering/Speed).
  * **LiDAR Projection:** Nghiên cứu cách chiếu (project) Point Cloud từ 3D xuống 2D (BEV) và lên khung hình Camera.
  * **Data Packaging:** Đóng gói các cặp [Video clip - LiDAR BEV - Action] vào định dạng dễ đọc (như `.h5` hoặc `.pt`) để nhóm có thể "lôi ra dùng ngay" khi có giải pháp.

### P2. Thích nghi Nhận thức

* **Vấn đề cần giải:** Làm sao để Encoder của V-JEPA (vốn học từ Mỹ/Singapore) nhận diện được các đặc trưng "hỗn loạn" như xe máy, vỉa hè không rõ ràng, hay các vật thể di động tại VN?
* **Nhiệm vụ:**
  * Nghiên cứu cấu hình **Masking Strategy** tối ưu: Với xe máy nhỏ và mật độ dày, nên che ảnh theo tỉ lệ nào để mô hình buộc phải học được sự tồn tại của chúng?
  * Thử nghiệm chạy Inference V-JEPA trên dữ liệu VF và dùng các phương pháp Visualization (tìm kiếm thêm các phương pháp) để xem mô hình đang "nhìn" vào cái gì trên đường phố VN.

### P3. Nghiên cứu "Hành vi giả lập"

* **Vấn đề cần giải:** Bộ 8192 quỹ đạo mẫu của Drive-JEPA (lái xe kiểu Mỹ) có thể sẽ không phù hợp ở VN.
* **Nhiệm vụ:**
  * Phân tích dữ liệu Steering/GPS của VF để tìm ra các hành vi đặc trưng: Lách xe máy, bám đuôi trong tắc đường, rẽ trái kiểu VN.
  * Nghiên cứu thuật toán **Clustering (K-means/GMM)** để tự động hóa việc tạo "Trajectory Vocabulary" từ chính dữ liệu thực tế tại Việt Nam thay vì dùng bộ mẫu có sẵn của bài báo.

### P4: Nghiên cứu Môi trường Đánh giá

* **Vấn đề cần giải:** Thay thế HD Map và 3D Bounding Box bằng cái gì để tính được PDMS?
* **Nhiệm vụ:**
  * Nghiên cứu giải pháp phát hiện va chạm chỉ bằng Lidar.
  * Tìm cách "lái lại" (Re-simulation) trên video thực tế mà không cần môi trường 3D.
  * Xây dựng công thức tính **PDMS** (equivalent to PDMS).

### P5. Nghiên cứu Hợp nhất

* **Vấn đề cần giải:** Làm sao để Planner học được từ bộ chấm điểm chỉnh sửa của P4 mà vẫn đạt hiệu quả cao?
* **Nhiệm vụ:**
  * Giống P4
  * Nghiên cứu cách huấn luyện để Planner không chỉ bắt chước tài xế mà còn biết tự tránh vật cản dựa trên LiDAR BEV.
