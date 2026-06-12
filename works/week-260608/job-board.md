# Phân chia công việc tuần 09/06/26 - 22/06/26

## 1. Overview

| Người phụ trách | Công việc |
| :--- | :--- |
| Ngọc + Phượng + Thành | Lit-Review |
| Phượng + Thành + Nam | Thu thập dữ liệu |
| Phượng + Khanh + Tuấn Anh | Đánh Giá 1 |
| Ngọc + Phượng + Nam | Đánh Giá 2 |
| Ngọc | Tìm hiểu các phương pháp đánh giá và cải tiến mô hình |
| Ngọc | Viết PhD Proposal |
| Thành | Yêu cầu 1 - Push code lên Azure DevOps |
| Khanh | Yêu cầu 2 - Cập nhật phần đánh giá trực quan hóa |

---

## Kế Hoạch Chi Tiết

### Tuần 1: Thu Thập Dữ Liệu & Đọc Hiểu Lý Thuyết

| Đối tượng | Công việc cụ thể | Sản phẩm đầu ra (Deliverables) |
| --- | --- | --- |
| **Nhóm Lit-Review** | Tìm và đọc tối thiểu 3-4 bài báo cốt lõi về: **V-JEPA2**, **Perception**, **World Models trong Autonomous Driving**. | Một file tổng hợp (Matrix Sheet): Phương pháp, Hàm Loss, Tập dữ liệu, và Điểm yếu cốt lõi của từng bài báo. |
| **Nhóm Thu Thập Dữ Liệu** | Tìm kiếm và thu thập dữ liệu từ các nguồn mở (YouTube, diễn đàn ô tô) để bổ sung vào tập test-case.<br>* Thu thập và phân loại các đoạn video/ảnh từ camera hành trình có **hạ tầng phi cấu trúc** (chợ tự phát, đường mất vạch) và **thực thể dị hình** (xe ba gác, xe lôi).<br>* Cắt các đoạn video hành trình ngắn (3-5 giây) có các tình huống **động lực học dị biệt**: xe máy tạt đầu cự ly hẹp, đi ngược chiều, lóa sáng, mưa loang lổ. | * Tập dữ liệu test-case tĩnh (khoảng 100-200 ảnh mẫu đặc trưng Việt Nam).<br>* Tập dữ liệu test-case động (khoảng 30-50 video clips ngắn, 3-5 giây, chất lượng cao). |
| **Nhóm Đánh Giá 1** | * Tìm hiểu cơ chế trích xuất Token của khối ViT.<br>* Đọc hiểu hàm Cosine Similarity áp dụng trên Feature patches.<br>* Tìm hiểu về các thuật toán đánh giá không gian ẩn nâng cao để đo độ lệch phân phối (OOD) (gợi ý: Mahalanobis Distance). | * Slide trình bày bản chất toán học của cơ chế trích xuất Token và đánh giá không gian.<br>* Đề xuất phương pháp đánh giá không gian (nếu có) |
| **Nhóm Đánh Giá 2** | * Tìm hiểu cách Predictor dự đoán trạng thái ẩn tương lai $z_{t+1}$ từ $z_t$.<br>* Tìm hiểu thêm: **Thông tin tương hỗ (Mutual Information)** hoặc các độ đo kiểm định tính ổn định chuỗi thời gian (*Temporal Consistency*). | * Slide trình bày về cơ chế dự đoán trạng thái ẩn và đánh giá tính ổn định chuỗi thời gian.<br>* Đề xuất phương pháp đánh giá tính ổn định chuỗi thời gian (nếu có) |

---

### Tuần 2: Chạy Thực Nghiệm, Trích Xuất Metrics & Tổng Hợp

| Đối tượng | Công việc cụ thể | Sản phẩm đầu ra (Deliverables) |
| --- | --- | --- |
| **Nhóm Lit-Review** | * Bổ sung 1-2 bài nếu cần.<br>* Viết bản thảo chương Tổng quan tài liệu. | Bản thảo văn bản chương Literature Review hoàn chỉnh (File Word/LaTeX). |
| **Nhóm Thu Thập Dữ Liệu** | * Tiếp tục thu thập bù nếu có test-case bị lỗi/thiếu.<br>* Tiền xử lý dữ liệu (chuyển đổi định dạng, chuẩn hóa, loại bỏ nhiễu) theo đúng yêu cầu đầu vào của V-JEPA2. | Bộ dữ liệu thử nghiệm chuẩn hóa (Dữ liệu Sạch): Được đóng gói thành cấu trúc thư mục hoàn chỉnh, sẵn sàng nạp thẳng vào code của Nhóm Đánh giá mà không bị lỗi runtime. |
| **Nhóm Đánh Giá 1** | * Hiện thực hóa các thuật toán đánh giá không gian (Cosine / Mahalanobis) bằng code Python trên nền môi trường V-JEPA2 có sẵn.<br>* Chạy thực nghiệm trên tập dữ liệu tĩnh sạch do Nhóm Data bàn giao. | * Bản đồ nhiệt chú ý (Spatial Attention Heatmaps) chồng lên ảnh gốc.<br>* Ma trận số liệu thể hiện độ lệch phân phối (OOD scores) của dữ liệu Việt Nam so với tập pre-trained. |
| **Nhóm Đánh Giá 2** | * Hiện thực hóa các thuật toán đánh giá tính ổn định chuỗi thời gian (Predictor Loss / Mutual Information) bằng code Python.<br>* Chạy thực nghiệm trên tập video động sạch do Nhóm Data bàn giao. | * Đồ thị đường (Line plots) thể hiện sự biến động của sai số dự đoán qua từng khung hình (chỉ rõ các điểm Loss vọt lên khi có tạt đầu/nhiễu thời tiết). |

### Yêu cầu từ Vinfast

1. Push code lên Azure DevOps:
    * Nhánh 1: `training` - chứa code mô hình huấn luyện và đánh giá (full pipeline)
    * Nhánh 2: `converter` - chứa code chuyển đổi dữ liệu từ định dạng VF sang định dạng của NAVSIM
2. Cập nhật phần đánh giá trực quan hóa:
    * Thêm đơn vị đo lường (mét) vào trục khi vẽ quỹ đạo. Khi vẽ quỹ đạo lên ảnh/video, cần bổ sung rõ các trục đơn vị đo lường (ví dụ: mét - biểu diễn khoảng cách tương đối).
    * Xây dựng video Ground Truth tích hợp cảm biến (camera + IMU + CAN-bus). Chèn các tín hiệu xử lý và tín hiệu từ cảm biến lên góc video để người xem biết lúc đó xe đang ở trạng thái vật lý nào (ví dụ: tốc độ, góc đánh lái, gia tốc).
