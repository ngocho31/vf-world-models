# Phân chia công việc tuần 23/06/26 - 06/07/26

## 1. Overview

| Người phụ trách | Công việc |
| :--- | :--- |
| Ngọc + Phượng + Thành | Lit-Review |
| Phượng + Thành + Nam | Thu thập dữ liệu |
| Khanh + Tuấn Anh | Đánh Giá 1 |
| Ngọc + Phượng + Nam | Đánh Giá 2 |
| Ngọc | Tìm hiểu các phương pháp đánh giá và cải tiến mô hình |
| Ngọc | Viết PhD Proposal |

---

## Kế Hoạch Chi Tiết

| Đối tượng | Công việc cụ thể | Sản phẩm đầu ra (Deliverables) |
| --- | --- | --- |
| **Nhóm Lit-Review** | Bổ sung 1-2 bài (nếu cần.) về **V-JEPA2**, **Perception**, **World Models trong Autonomous Driving**. | Một file tổng hợp (Matrix Sheet): Phương pháp, Hàm Loss, Tập dữ liệu, Điểm yếu cốt lõi của từng bài báo và đề xuất cải tiến (nếu có). |
| **Nhóm Thu Thập Dữ Liệu** | * Tiếp tục thu thập bù nếu có test-case bị lỗi/thiếu.<br>* Tiền xử lý dữ liệu (chuyển đổi định dạng, chuẩn hóa, loại bỏ nhiễu) theo đúng yêu cầu đầu vào của V-JEPA2. | Bộ dữ liệu thử nghiệm chuẩn hóa (Dữ liệu Sạch): Được đóng gói thành cấu trúc thư mục hoàn chỉnh, sẵn sàng nạp thẳng vào code của Nhóm Đánh giá mà không bị lỗi runtime. |
| **Nhóm Đánh Giá 1** | * Hiện thực hóa thuật toán tính toán Cosine Similarity giữa các cặp token không gian.<br>* Chạy thực nghiệm trên tập dữ liệu của openscene (https://huggingface.co/datasets/OpenDriveLab/OpenScene/tree/main/navsim).<br>* Chạy thực nghiệm trên tập dữ liệu của Nhóm Data bàn giao. | * Bản đồ nhiệt chú ý (Spatial Attention Heatmaps) chồng lên ảnh gốc.<br>* Ma trận số liệu thể hiện điểm số Cosine Similarity của dữ liệu Việt Nam so với tập pre-trained. |
| | * Hiện thực hóa thuật toán tính toán khoảng cách Mahalanobis.<br>* Chạy thực nghiệm trên tập dữ liệu của openscene và tập dữ liệu của Nhóm Data bàn giao. | * Bản đồ nhiệt bất thường (Anomaly Attention Heatmaps) chồng lên ảnh gốc.<br>* Ma trận số liệu thể hiện điểm số Mahalanobis của dữ liệu Việt Nam so với tập pre-trained. |
| **Nhóm Đánh Giá 2** | * Tìm hiểu cách Predictor dự đoán trạng thái ẩn tương lai $z_{t+1}$ từ $z_t$.<br>* Tìm hiểu thêm: **Thông tin tương hỗ (Mutual Information)** hoặc các độ đo kiểm định tính ổn định chuỗi thời gian (*Temporal Consistency*). | * Slide trình bày về cơ chế dự đoán trạng thái ẩn và đánh giá tính ổn định chuỗi thời gian.<br>* Đề xuất phương pháp đánh giá tính ổn định chuỗi thời gian (nếu có) |
| | * Hiện thực hóa các thuật toán đánh giá tính ổn định chuỗi thời gian (Predictor Loss / Mutual Information) bằng code Python.<br>* Chạy thực nghiệm trên tập video động sạch do Nhóm Data bàn giao. | * Đồ thị đường (Line plots) thể hiện sự biến động của sai số dự đoán qua từng khung hình (chỉ rõ các điểm Loss vọt lên khi có tạt đầu/nhiễu thời tiết). |
