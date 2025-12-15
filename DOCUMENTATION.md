# Zero-Shot Depth-Aware Image Editing

---

## 1. Phân tích Bài báo (Paper Analysis)

**Dựa trên:** Paper "Zero-Shot Depth-Aware Image Editing with Diffusion Models" (ICCV 2025)

### A. Abstract (Tóm tắt)

#### **Vấn đề (Problem)**
- Các mô hình Diffusion hiện nay sửa ảnh rất tốt nhưng **gặp khó khăn trong việc hiểu không gian 3D** (độ sâu)
- **Ví dụ:** Việc đặt một vật thể (cái ghế) ra sau cái bàn nhưng phải nằm trước bức tường là rất khó thực hiện chỉ bằng text prompt

#### **Giải pháp (Solution)**
Phương pháp Zero-shot (không cần huấn luyện lại mô hình) bao gồm 2 bước chính:

1. **Depth-Guided Layer Decomposition (DeGLaD)**
   - Tự động tách ảnh thành các lớp (Tiền cảnh & Hậu cảnh)
   - Dựa trên giá trị độ sâu do người dùng chỉ định

2. **Feature Guided Layer Compositing (FeatGLaC)**
   - Trộn các lớp ngay trong không gian đặc trưng (feature space) của U-Net
   - Thay thế việc cắt dán pixel thô thiển
   - Giúp các layer hòa trộn tự nhiên hơn

#### **Kết quả (Output)**
✅ Tạo ra ảnh ghép vật thể/cảnh tuân thủ quy luật 3D  
✅ Ánh sáng hài hòa tự nhiên  
✅ Không cần mô hình hậu kỳ (harmonization)

### B. Kết luận (Conclusion)

#### **Ưu điểm**
- Vượt trội hơn các kỹ thuật cắt dán (copy-paste) truyền thống
- Vượt trội hơn các mô hình inpainting thông thường
- **Bảo toàn cấu trúc 3D** của cảnh quan

#### **Hạn chế**
- Phụ thuộc vào độ chính xác của mô hình ước lượng độ sâu (MiDaS/ZoeDepth)
- Nếu Depth Map sai → Kết quả chỉnh sửa cũng sai lệch

---

## 2. Bản chất Mô hình & Hàm mất mát (Model Nature & Loss Function)

### A. Bản chất mô hình (Nature of the Model)

Đây **KHÔNG phải** một mô hình đơn lẻ mà là một **Pipeline** kết hợp sức mạnh của nhiều mô hình Pre-trained:

#### **1. Backbone (Xương sống): Latent Diffusion Model (LDM)**
- Thường là **Stable Diffusion**
- **Bản chất:** Mô hình sinh xác suất
- **Chức năng:** Học cách loại bỏ nhiễu (denoising) dần dần để tạo ra ảnh rõ nét
- **Tham số:** ~1.5B (khá lớn)

#### **2. Depth Estimator (Nhận thức độ sâu)**
- Sử dụng: **MiDaS** hoặc **ZoeDepth**
- **Bản chất:** Các mạng CNN/Transformer chuyên biệt
- **Chức năng:** Dự đoán khoảng cách của từng pixel từ ảnh 2D
- **Đầu ra:** Depth Map (bản đồ độ sâu)

#### **3. Identity Preserver (Giữ nhân dạng vật thể)**
- Sử dụng: **AnyDoor** (hoặc module tương tự)
- **Bản chất:** Reference-guided Generation
- **Chức năng:** Mã hóa ảnh vật thể tham chiếu thành feature vector và "tiêm" vào quá trình sinh ảnh
- **Kết quả:** Vật thể không bị biến dạng

### B. Hàm mất mát (Loss Function)

> ⚠️ **Lưu ý:** Vì đây là phương pháp Zero-shot (Inference-only), **không thực hiện quá trình huấn luyện** nên sẽ không trực tiếp tối ưu hóa hàm loss nào trong lúc chạy code.

Tuy nhiên, để hiểu **tại sao nó hoạt động**, các mô hình nền tảng đã được huấn luyện trước đó với các hàm loss sau:

#### **1. Noise Prediction Loss (MSE)**
$$\mathcal{L}_{denoise} = \mathbb{E}_{x_0, t, \epsilon} [\|\epsilon - \epsilon_\theta(x_t, t)\|_2^2]$$

- **Ý nghĩa:** Máy học cách dự đoán xem lớp nhiễu nào đã được thêm vào ảnh tại thời điểm $t$

#### **2. Perceptual Loss (LPIPS)**
$$\mathcal{L}_{perceptual} = \sum_l \frac{1}{N_l} \sum_{h,w} \|F_l(x) - F_l(y)\|_2^2$$

- Thường dùng trong AnyDoor/Autoencoder
- **Ý nghĩa:** Đảm bảo ảnh sinh ra nhìn "thật" và giống ảnh gốc về mặt tri giác của mắt người

#### **3. Feature Matching Loss**
$$\mathcal{L}_{feat-match} = \|F_{ref} - F_{generated}\|_2$$

- **Ý nghĩa:** Ép buộc đặc trưng của vật thể được ghép khớp với đặc trưng của vật thể mẫu ban đầu


---

## 3. Kiến Trúc Hệ Thống (System Architecture)

Hệ thống được chia làm **2 Module chính** xử lý tuần tự:

```
DeGLaD (Tách) ──→ FeatGLaC (Ghép)
```

### Module 1: DeGLaD (Depth-Guided Layer Decomposition)

**Chức năng:** Tách ảnh đầu vào thành các lớp không gian (Layers) dựa trên độ sâu

**Logic xử lý:**

1. Dùng mô hình (MiDaS) **ước lượng độ sâu** ảnh
2. Người dùng **chọn ngưỡng độ sâu $d$** (ví dụ: vị trí cái bàn)
3. **Tách ảnh thành:**
   - **Tiền cảnh:** Độ sâu < $d$ (gần hơn)
   - **Hậu cảnh:** Độ sâu > $d$ (xa hơn)
4. **Quan trọng:** Khi tách Tiền cảnh ra, Hậu cảnh sẽ bị **thủng một lỗ**
   - Module này tự động dùng **AI Inpainting** để vẽ bù vào lỗ đó

### Module 2: FeatGLaC (Feature-Guided Layer Compositing)

**Chức năng:** Ghép các lớp lại thành ảnh hoàn chỉnh sao cho ánh sáng và bóng đổ **tự nhiên**

**Logic xử lý:**

1. **KHÔNG ghép chồng pixel** (Alpha Blending) vì sẽ lộ viền
2. **Sử dụng cơ chế Feature Injection** trong mạng U-Net
3. **Guidance Branch:** Chạy song song để trích xuất cấu trúc $(K, V)$ từ các lớp Tiền/Hậu cảnh
4. **Generation Branch:** 
   - Sinh ảnh mới
   - Bị ép buộc phải tuân theo cấu trúc $(K, V)$ của nhánh hướng dẫn


---

## 4. Ứng Dụng vào Dự Án Big Data (Implementation)

Phần này mô tả việc **mở rộng** mô hình trên (chỉ chạy 1 ảnh) thành một **dây chuyền xử lý 10GB dữ liệu** trên Kaggle.

### Quy Trình Xử Lý Batch (Batch Processing Pipeline)

Sử dụng kiến trúc **Data Parallelism** (Song song dữ liệu)

#### **Bước 1: Thu thập & Tiền xử lý (Storage)**

- **Thay vì tải ảnh lẻ tẻ**, nhóm sử dụng **Video 4K tự quay**
- Dùng script Python (OpenCV) **cắt video thành 25,000 ảnh tĩnh**
- **Lợi ích:**
  - ✅ Dữ liệu lớn
  - ✅ Chất lượng cao
  - ✅ Đồng nhất

#### **Bước 2: Phân phối (Controller)**

- Dữ liệu 10GB **không thể nạp hết vào RAM**
- Sử dụng **DataLoader** để nạp cuốn chiếu (Streaming):
  - Nạp 32 ảnh → Xử lý → Xóa khỏi RAM → Nạp 32 ảnh tiếp

#### **Bước 3: Xử lý Song song (Workers)**

- **Môi trường:** Kaggle cung cấp **2 GPU T4**
- **Thuật toán:** DataParallel tự động chia đôi Batch
  - 16 ảnh → GPU 1
  - 16 ảnh → GPU 2
- **Thực hiện:** Mỗi GPU thực hiện trọn vẹn **DeGLaD + FeatGLaC** cho phần ảnh của mình
- **Tốc độ:** ~2.5s/ảnh × 25,000 ảnh ÷ 2 GPUs ≈ **14 giờ** (với overhead)

#### **Bước 4: Tổng hợp (Reducer)**

- **Kết quả output** được ghi liên tục xuống ổ cứng
- Sau khi chạy xong toàn bộ:
  - Script cuối cùng quét toàn bộ ảnh kết quả
  - Tính toán **sai số (RMSE)**
  - Vẽ biểu đồ **đánh giá kết quả**

---

## 📊 Tóm Tắt Pipeline

| Bước | Thành Phần | Chức Năng |
|------|-----------|----------|
| 1 | Video Input | Quay 4K video gốc |
| 2 | Frame Extraction | Cắt video → 25,000 ảnh |
| 3 | Depth Estimation | MiDaS ước lượng độ sâu |
| 4 | DeGLaD | Tách lớp theo depth |
| 5 | Inpainting | Vẽ bù lỗ thủng |
| 6 | FeatGLaC | Ghép lớp mượt mà |
| 7 | Output | Ảnh chỉnh sửa 3D-aware |
| 8 | Evaluation | RMSE + Biểu đồ |


