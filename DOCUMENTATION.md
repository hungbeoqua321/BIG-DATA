# BÁO CÁO KỸ THUẬT: ZERO-SHOT DEPTH-AWARE IMAGE EDITING
**Chỉnh sửa ảnh nhận biết độ sâu không cần huấn luyện lại**

Nguồn tài liệu: Parihar et al., "Zero-Shot Depth-Aware Image Editing with Diffusion Models", ICCV 2025.

---

## 0. BẢNG THUẬT NGỮ & ĐỊNH NGHĨA

| Thuật ngữ | Giải thích chi tiết |
|-----------|-------------------|
| **Zero-Shot** | Khả năng thực hiện tác vụ mới mà không cần huấn luyện lại. AI có thể dùng ngay trên dữ liệu chưa từng thấy mà vẫn hoạt động tốt. |
| **Diffusion Models** | Mô hình khuếch tán - phương pháp sinh ảnh bằng cách học khử nhiễu. Bắt đầu từ ảnh nhiễu hoàn toàn, dần dần khôi phục lại ảnh rõ nét. Ví dụ: Stable Diffusion. |
| **Depth-Aware** | Nhận thức độ sâu - khả năng hiểu không gian 3D (trục Z). Máy biết vật nào ở gần, vật nào ở xa. |
| **Occlusion** | Sự che khuất - hiện tượng vật ở gần che khuất vật ở xa. Xử lý đúng occlusion là yếu tố sống còn để ảnh ghép tự nhiên. |
| **Inpainting** | Vẽ bù/Điền khuyết - kỹ thuật dùng AI tự động vẽ lại phần hình ảnh bị mất hoặc bị che khuất. |
| **Latent Space** | Không gian tiềm ẩn - dữ liệu ảnh được nén thành vector đặc trưng để xử lý nhanh hơn. |
| **Self-Attention** | Cơ chế tự chú ý - cách AI xác định mối quan hệ giữa các điểm ảnh. |
| **Feature Injection** | Tiêm đặc trưng - can thiệp vào mô hình để ép nó tuân theo cấu trúc mong muốn. |
| **Data Parallelism** | Song song dữ liệu - chia dữ liệu thành nhiều phần để xử lý trên nhiều GPU cùng lúc. |

---

## 0.1 CHI TIẾT: LATENT SPACE (Không gian tiềm ẩn)

### Khái niệm cơ bản

Hình ảnh bình thường ở **pixel space:**
- Ảnh 512×512×3 = **786,432 giá trị pixel** (R, G, B)
- Mỗi giá trị 0-255
- Khó học vì dữ liệu quá lớn

**Latent Space** = không gian nén lại:
- Ảnh 512×512 → nén thành **64×64×4 = 16,384 giá trị** (roughly 48 lần nhỏ hơn)
- Giữ lại thông tin quan trọng, bỏ chi tiết không cần
- Mô hình học nhanh, tiêu tốn memory ít

### Cách hoạt động

**VAE Encoder (Mã hóa):**
```
Ảnh 512×512×3 (786K values)
    ↓
CNN layers (conv, pooling, residual blocks)
    ↓
Bottleneck layer → gaussian distribution
    ↓
Latent vector 64×64×4 (16K values)
```

**VAE Decoder (Giải mã):**
```
Latent vector 64×64×4
    ↓
Transposed CNN (upsampling, conv)
    ↓
Ảnh 512×512×3
```

### Tại sao quan trọng?

1. **Tốc độ:** Diffusion model tạo tiếng ồn trong latent space (nhanh 8x)
2. **Memory:** Lưu 16K giá trị thay vì 786K (tiết kiệm VRAM)
3. **Chất lượng:** Mô hình học đặc trưng thay vì chi tiết pixel vô nghĩa
4. **Linh hoạt:** Có thể dùng cùng một latent space cho nhiều task

**Ví dụ thực tế:**
- Stable Diffusion: sử dụng VAE để làm việc ở latent space
- Khiến mô hình chỉ cần ~1.5 tỷ tham số thay vì 10+ tỷ

---

## 0.2 CHI TIẾT: SELF-ATTENTION (Cơ chế tự chú ý)

### Khái niệm cơ bản

**Self-Attention** = cách AI xác định "điểm ảnh nào quan trọng với nhau"

Ví dụ trong câu tiếng Anh:
```
"The dog saw the cat and it ran away"
     ↑                        ↑
  Từ "it" nên chú ý tới từ "dog" (không phải "cat")
```

Tương tự, trong ảnh:
```
Khi vẽ chi tiết mắt mèo → chú ý tới pixel mắt + vùng xung quanh
Không cần chú ý tới background (cây, bầu trời)
```

### Cơ chế toán học

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Ba thành phần:**

1. **Query (Q):** "Tôi đang xử lý pixel nào?"
2. **Key (K):** "Pixel khác nào liên quan?"
3. **Value (V):** "Lấy thông tin gì từ pixel liên quan?"

**Ví dụ cụ thể:**

```
Input: Ảnh 64×64×256 channels (từ CNN layer)

Step 1: Tạo Q, K, V từ feature map
  Q = Linear_Q(feature_map)    # 4096×256 → 4096×64
  K = Linear_K(feature_map)    # 4096×256 → 4096×64
  V = Linear_V(feature_map)    # 4096×256 → 4096×256

Step 2: Tính sự tương đồng
  scores = Q @ K^T             # 4096×4096 (mỗi pixel với tất cả pixel khác)

Step 3: Chuẩn hóa
  weights = softmax(scores)    # Tổng bằng 1 mỗi dòng

Step 4: Trích thông tin
  output = weights @ V         # 4096×256 (thông tin có trọng số)
```

### Tại sao quan trọng trong bài toán này?

**Trong FeatGLaC** (Feature-Guided Layer Compositing):

```
Original U-Net Self-Attention:
  Attention(Q, K, V) 
  ↓
  Tạo ảnh dựa vào noise + diffusion timestep

Modified (Feature Injection):
  Attention(Q, K_guided, V_guided)
  ↓
  Ép buộc tạo ảnh tuân theo cấu trúc layers được tách
```

**Kết quả:** 
- Model tự động điều chỉnh ánh sáng, bóng đổ
- Không cần hard merge giữa tiền/hậu cảnh
- Ánh sáng tự nhiên vì diffusion process tự khôi phục harmonic

---

## 0.3 CHI TIẾT: FEATURE INJECTION (Tiêm đặc trưng)

### Khái niệm cơ bản

**Feature Injection** = can thiệp vào quá trình tính toán của mô hình để ép nó tuân theo một cấu trúc

Ví dụ trong đời sống:
```
Bình thường: Máy tự vẽ ảnh tự do
Với injection: "Hãy vẽ nhưng bắt buộc phải có cái cây ở góc trái"
```

### Cách hoạt động trong FeatGLaC

**Bước 1: Chuẩn bị Guidance Features**

```
Tiền cảnh Layer + Hậu cảnh Layer
    ↓
VAE Encoder → Latent representations
    ↓
Guidance U-Net (mô hình phụ)
    ↓
Trích features từ mỗi attention layer
    K_fg, V_fg = from foreground
    K_bg, V_bg = from background
```

**Bước 2: Injection vào Generation U-Net**

```
Diffusion loop (t = T đến 0):
    ↓
  noise_t = model(z_t, t, text_prompt)  ← bình thường
  ↓
  Tại mỗi Self-Attention layer:
    
    Q_gen = from generation U-Net (tính bình thường)
    K_gen, V_gen = bình thường
    
    Thay thế:
    K_gen ← blend(K_fg, K_bg)  ← dùng guidance features!
    V_gen ← blend(V_fg, V_bg)
    
    output = Attention(Q_gen, K_gen_injected, V_gen_injected)
```

**Bước 3: Kết quả**

```
output = DecoderUNet(...)
    ↓
VAE Decoder → ảnh cuối cùng
    ↓
Ảnh với:
  ✅ Cấu trúc giống tiền/hậu cảnh
  ✅ Ánh sáng tự động cân chỉnh
  ✅ Chi tiết mượt mà (không cạnh cứng)
```

### Tại sao hoạt động tốt?

1. **Self-Attention** học "mối quan hệ toàn cảnh"
   - Biết ánh sáng nên từ đâu
   - Biết bóng đổ nên ở chỗ nào
   
2. **Diffusion Process** = lặp 50 bước khử nhiễu
   - Không làm một lần (kiểu hard merge)
   - Từ từ điều chỉnh, các chi tiết không nhất quán tự khôi phục

3. **Feature Injection** = soft constraint
   - Không bắt buộc tuân theo 100% (sẽ cứng, giả tạo)
   - Ép buộc nhưng để mô hình linh hoạt điều chỉnh

### Ví dụ thực tế

**Chèn ghế vào ảnh phòng:**

```
Tiền cảnh: Cái bàn
Hậu cảnh: Tường, cửa, sàn

Bình thường (hard merge):
  Ghế dán lên bàn → nó trông sáng bất thường (không bóng)

Với Feature Injection:
  1. Guidance features nói: "Đây là vùng bàn, vùng tường"
  2. Generation U-Net sinh ghế nhưng inject guidance
  3. Self-Attention layers tự động:
     - Ghế ở phía sau bàn (occlusion)
     - Ánh sáng từ cửa → ghế tối phía một
     - Bóng đổ hợp lý trên sàn
  4. Kết quả: tự nhiên, không cần post-processing
```

---

## 1. PHÂN TÍCH VẤN ĐỀ (PROBLEM STATEMENT)

### 1.1 Vấn đề hiện tại

Các công cụ chỉnh sửa ảnh AI (Photoshop Generative Fill, inpainting) **hoạt động trên mặt phẳng 2D** - không hiểu quan hệ không gian 3D.

**Ví dụ cụ thể:**
- Khi chèn cái ghế vào ảnh: dán ghế đè lên mọi thứ (không biết ghế nằm sau bàn)
- Ánh sáng không tự nhiên (ghế sáng trong khi bàn tối)
- Không tạo bóng đổ phù hợp

**Hậu quả:** Ảnh nhìn giả tạo, sai phối cảnh, mất tính logic vật lý.

### 1.2 Giải pháp

Bài báo ICCV 2025 đề xuất quy trình **DeGLaD + FeatGLaC**:
1. **DeGLaD:** Tách ảnh thành các lớp (tiền cảnh/hậu cảnh) dựa trên độ sâu
2. **FeatGLaC:** Ghép lại với ánh sáng tự nhiên bằng Feature Injection

**Tất cả đều Zero-Shot:** Không cần huấn luyện mô hình mới.

### 1.3 Kết quả mong đợi

✅ Ảnh ghép tuân thủ quy luật 3D  
✅ Ánh sáng và bóng đổ tự nhiên, hài hòa  
✅ Không cần quá trình hậu xử lý phức tạp

---

## 2. CÁC THÀNH PHẦN CỐT LÕI (CORE COMPONENTS)

Hệ thống là **pipeline kết hợp** nhiều mô hình pre-trained, không phải mô hình đơn lẻ.

### 2.1 Latent Diffusion Model (LDM) - Xương sống

**Bản chất:** Mô hình sinh (Generative Model) xác suất

**Mô hình cụ thể:** Stable Diffusion hoặc biến thể

**Cơ chế:**
- **Quá trình Khuếch tán:** Thêm nhiễu vào ảnh gốc từ từ
- **Quá trình Khử Nhiễu:** Học cách loại bỏ nhiễu để khôi phục ảnh rõ nét
- **Latent Space:** Hoạt động trên không gian đặc trưng (4-8 lần nhỏ hơn pixel)

**Tham số:** ~1.5 tỷ

**Lợi ích:**
- Khả năng sinh ảnh đa dạng
- Chất lượng cao, chi tiết đẹp
- Có thể hướng dẫn bằng text hoặc image guidance

### 2.2 Depth Estimator - Cảm biến 3D

**Bản chất:** Mạng CNN/Transformer chuyên biệt dự đoán độ sâu

**Mô hình:** MiDaS hoặc ZoeDepth

**Chức năng:**
- Input: Ảnh 2D (RGB)
- Output: Depth Map (giá trị 0-255 hoặc 0-1)
  - Giá trị cao = xa (nền)
  - Giá trị thấp = gần (tiền cảnh)

**Độ chính xác:** Tương đối tốt trên ảnh thực tế

**Giới hạn:**
- Là relative depth, không metric depth
- Có thể sai trên vật thể trong suốt, bóng đổ

### 2.3 Identity Preserver (AnyDoor) - Giữ nhân dạng

**Bản chất:** Mô hình Reference-Guided Generation

**Chức năng:** Đảm bảo vật thể không bị biến dạng, giữ nguyên nhân dạng gốc

**Cơ chế:**
1. **Encoding:** Mã hóa ảnh tham chiếu thành feature vector $F_{ref}$
2. **Injection:** Tiêm $F_{ref}$ vào quá trình sinh ảnh của U-Net
3. **Ép buộc:** U-Net phải sử dụng $F_{ref}$, nên kết quả giữ nguyên nhân dạng

**Lợi ích:**
- Vật thể không méo mó, biến hình
- Giữ lại chi tiết gốc
- Hỗ trợ vật phức tạp

### 2.4 VAE Encoder/Decoder - Cầu nối

**Bản chất:** Variational Autoencoder

**Chức năng:**
- **Encoder:** Ảnh (hàng triệu giá trị) → latent space (hàng trăm giá trị)
- **Decoder:** Latent space → ảnh để hiển thị

**Tại sao cần:**
- Giảm chi phí tính toán 4-8 lần
- Tập trung vào đặc trưng quan trọng
- Tăng tốc độ sinh ảnh

---

## 3. KIẾN TRÚC HỆ THỐNG (SYSTEM ARCHITECTURE)

Hệ thống hoạt động theo quy trình **tuần tự 2 bước chính.**

### 3.1 Bước 1: DeGLaD (Depth-Guided Layer Decomposition)

**Chức năng:** Tách ảnh thành các lớp không gian dựa trên độ sâu

#### Quy trình chi tiết

**Input:**
- Ảnh gốc (RGB)
- Bản đồ độ sâu (từ MiDaS)
- Ngưỡng độ sâu $d$ (chọn bởi người dùng)

**Bước 1: Ước lượng Depth Map**
```
depth_map = MiDaS(rgb_image)  # Kết quả 0-1 hoặc 0-255
```

**Bước 2: Tạo mặt nạ**
```
mask_foreground = depth_map < d       # Pixels gần hơn ngưỡng
mask_background = depth_map >= d      # Pixels xa hơn ngưỡng
```

**Bước 3: Tách lớp**
```
layer_fg = rgb_image * mask_foreground
layer_bg = rgb_image * mask_background
```

**Bước 4: Inpainting lỗ thủng**
- Lớp Hậu cảnh có "lỗ đen" ở nơi Tiền cảnh che khuất
- Kích hoạt mô hình Inpainting để vẽ bù
- Sử dụng Diffusion inpainting hoặc CNN inpainting

**Output:**
- Layer Tiền cảnh (sạch, sẵn sàng chỉnh sửa)
- Layer Hậu cảnh (sạch, không lỗ)

#### Ưu điểm

✅ **Đơn giản:** Chỉ cần thresholds  
✅ **Nhanh:** Toàn xử lý nguyên lý học  
✅ **Điều chỉnh dễ:** Người dùng chọn ngưỡng

#### Hạn chế

❌ **Phụ thuộc Depth Map:** Nếu depth sai → kết quả sai  
❌ **Cạnh cứng:** Ranh giới tiền/hậu cảnh bị sắc  
❌ **Occlusion edge:** Khó xử lý cạnh mỏng, tóc

### 3.2 Bước 2: FeatGLaC (Feature-Guided Layer Compositing)

**Chức năng:** Ghép các lớp lại thành ảnh hoàn chỉnh với ánh sáng tự nhiên

#### Vấn đề với Alpha Blending

Ghép chồng pixel đơn giản:
```
output = fg * alpha + bg * (1 - alpha)
```

**Kết quả tồi:**
- Viền sắc ngoặc
- Ánh sáng không ăn nhập
- Bóng đổ không tự nhiên

#### Giải pháp: Feature Injection

**Thay vì ghép pixel, can thiệp vào bộ não (U-Net)** của mô hình Diffusion

**Kiến trúc hai nhánh:**

**Nhánh 1 - Guidance Branch:**
```
input_layers = [layer_fg, layer_bg]
↓
VAE Encoder → Latent vectors
↓
Guidance U-Net → xử lý
↓
Internal Features: K (Key), V (Value)
  K = cấu trúc hình học
  V = thông tin màu sắc
```

**Nhánh 2 - Generation Branch:**
```
noise_z ~ N(0, 1)
↓
Generation U-Net (T bước khử nhiễu)
  At each step t:
    Inject K, V vào Self-Attention layers
    → Ép buộc sinh ảnh tuân theo cấu trúc guidance
↓
VAE Decoder → Ảnh cuối cùng
```

#### Tại sao hoạt động tốt?

1. **K (Geometric Structure):** Cấu trúc ảnh sinh giống hệt layers đã tách
2. **V (Appearance):** Giữ lại màu sắc và chi tiết gốc
3. **Self-Attention Injection:** Ánh sáng AI tự động cân chỉnh (không pixel cứng nhắc)
4. **Diffusion Process:** Qua nhiều bước, các không nhất quán được giải quyết tự nhiên

#### Công thức toán học

$$\text{Attention}(Q, K_{\text{guided}}, V_{\text{guided}}) = \text{softmax}\left(\frac{QK_{\text{guided}}^T}{\sqrt{d}}\right)V_{\text{guided}}$$

Thay thế $K$ và $V$ gốc bằng các từ nhánh hướng dẫn.

---

## 4. TRIỂN KHAI HỆ THỐNG BIG DATA (IMPLEMENTATION)

Cách áp dụng thuật toán để xử lý **10GB dữ liệu** (~25,000 ảnh) trên Kaggle.

### 4.1 Thách thức & Khắc phục

| Thách thức | Nguyên nhân | Giải pháp |
|-----------|----------|---------|
| 10GB không vào RAM | RAM 32GB, CPU cũng cần không gian | Streaming DataLoader - batch 32 ảnh |
| Xử lý tuần tự quá lâu | 1 GPU × 2.5s × 25K = 70+ giờ | Data Parallelism - 2 GPU T4 |
| Mô hình 1.5B params | GPU 16GB VRAM, mô hình cần ~6GB | Batch nhỏ (32), Mixed Precision FP16 |
| Lưu 25K ảnh output | Dung lượng ~20-30GB | Write liên tục, không giữ RAM |

### 4.2 Kiến trúc Batch Processing Pipeline

#### Lớp 1: Data Storage

**Input gốc:** Video 4K tự quay

**Tiền xử lý:**
1. Cắt video thành frame (1 frame/0.1s)
2. Kết quả: ~25,000 ảnh PNG/JPEG
3. Lợi ích: Dữ liệu lớn, chất lượng cao, đồng nhất

#### Lớp 2: Data Controller

**Chức năng:** Quản lý luồng dữ liệu disk → RAM → GPU

**Thành phần:**

1. **File Manager:** Quét 25,000 ảnh, tạo danh sách
2. **DataLoader:** Nạp cuốn chiếu
   - 1 batch = 32 ảnh vào RAM
   - Sau xử lý, xóa khỏi RAM → nạp batch tiếp
   - Memory: ~500MB-1GB/lúc
3. **Preprocessing:**
   - Resize về 512×512 hoặc 768×768
   - Normalize (0-1 hoặc -1 to 1)
   - Chuẩn bị tensor

**Code logic:**
```
for batch in DataLoader(images, batch_size=32):
    process(batch)
    save_results(batch)
    del batch  # Giải phóng RAM
```

#### Lớp 3: Compute Cluster

**Cấu hình:**
- 2 × NVIDIA T4 (16GB VRAM mỗi)
- 8-core CPU
- 32GB system RAM

**Chia việc:**
```
batch = [img1, img2, ..., img32]
  ↓
Split:
  Part 1 (16 ảnh) → GPU 0
  Part 2 (16 ảnh) → GPU 1
  ↓
GPU 0 & GPU 1: DeGLaD + FeatGLaC (song song)
  ↓
Synchronize → kết quả gom
```

**Tốc độ:**
- 1 ảnh = ~2.5 giây
- 2 GPU song song = ~1.25 giây/ảnh (lý tưởng)
- Thực tế: ~1.8 giây/ảnh
- **Tổng:** 25,000 × 1.8s ÷ 3600 ≈ **12-14 giờ**

**Workflow trên mỗi GPU:**
```
for each image in batch:
    1. Load image → tensor
    
    2. DeGLaD:
       - Depth = MiDaS(image)
       - Mask_fg = depth < threshold
       - Layer_fg, Layer_bg = separate(image, mask)
       - Layer_bg = Inpaint(layer_bg)
    
    3. FeatGLaC:
       - Guidance_feats = Guidance_UNet(layer_fg, layer_bg)
       - Output = Generation_UNet_with_Injection(guidance_feats)
    
    4. Save output
```

#### Lớp 4: Aggregation & Evaluation

**Đầu vào:** 25,000 ảnh output

**Bước 1: Validation**
```
Check file size, format, dimension
Skip corrupted files
```

**Bước 2: Metric Calculation (RMSE)**
```
For each image:
    predicted_depth = MiDaS(output_image)
    true_depth = ground_truth[image_id]
    rmse = sqrt(mean((predicted - true)^2))
    
Final_RMSE = mean(all_rmse)
```

**Bước 3: Visualization**
- Biểu đồ RMSE qua image
- Histogram RMSE
- Top-K worst/best cases

### 4.3 Cấu hình Phần cứng & Tối ưu

| Thành phần | Chi tiết | Tác dụng |
|-----------|---------|---------|
| **GPU** | 2× T4 (16GB VRAM) | Xử lý DNN |
| **CPU** | 8-core Xeon | Data loading |
| **RAM** | 32GB DDR4 | Buffer |
| **Storage** | 100GB NVMe SSD | Input/Output |
| **Batch Size** | 32 | Cân bằng memory/throughput |
| **DataLoader Workers** | 8 | Parallel loading |
| **Mixed Precision** | FP16 | Giảm memory 2×, tăng tốc 1.5-2× |
| **Processing Time** | ~12-14 giờ | Thời gian thực tế |

### 4.4 Tóm tắt luồng dữ liệu

```
1. Video 4K
   ↓
2. Frame Extraction (25,000 ảnh)
   ↓
3. DataLoader (batch=32)
   ↓
4. GPU-0: DeGLaD+FeatGLaC (16) | GPU-1: DeGLaD+FeatGLaC (16)
   ↓                             ↓
5. Merge results
   ↓
6. Save to SSD (~20-30GB)
   ↓
7. Metric Calculation (RMSE)
   ↓
8. Report & Visualization
```

---

## 5. HÀM MẤT MÁT & HỌC (LOSS FUNCTIONS)

> **Lưu ý:** Zero-Shot (Inference-Only) → không training trực tiếp. Tuy nhiên mô hình pre-train với:

### 5.1 Noise Prediction Loss (MSE)

$$\mathcal{L}_{denoise} = \mathbb{E}_{x_0, t, \epsilon} \left[\|\epsilon - \epsilon_\theta(x_t, t)\|_2^2\right]$$

**Ý nghĩa:** Mô hình học dự đoán lớp nhiễu được thêm vào ảnh tại thời điểm $t$.

### 5.2 Perceptual Loss (LPIPS)

$$\mathcal{L}_{perceptual} = \sum_l \frac{1}{N_l} \sum_{h,w} \|F_l(x) - F_l(y)\|_2^2$$

**Ý nghĩa:** So sánh ở mức đặc trưng (không phải pixel). Đảm bảo ảnh sinh "thật" theo tri giác con người.

### 5.3 Feature Matching Loss

$$\mathcal{L}_{feat-match} = \|F_{ref} - F_{generated}\|_2$$

**Ý nghĩa:** Ép buộc đặc trưng vật thể giữ nguyên, không méo mó.

### 5.4 Reconstruction Loss (L1/L2)

$$\mathcal{L}_{recon} = \|x_{0} - \hat{x}_{0}\|_1$$

**Ý nghĩa:** Ảnh khôi phục gần với ảnh gốc.

---

## 6. TÓM TẮT & KỲ VỌNG

### Ưu điểm

✅ **Zero-Shot:** Không cần huấn luyện mô hình mới  
✅ **3D-Aware:** Xử lý occlusion, phối cảnh đúng  
✅ **Ánh sáng tự nhiên:** Feature Injection tự động điều chỉnh  
✅ **Mở rộng:** Áp dụng trên các tác vụ khác

### Hạn chế

❌ **Phụ thuộc depth map:** Depth sai → kết quả sai  
❌ **Tốc độ:** 2.5s/ảnh  
❌ **Tài nguyên:** GPU ≥16GB VRAM  
❌ **Chi tiết nhỏ:** Có thể lose tóc, cạnh mỏng

### Hướng phát triển

🔮 **Depth Improvement:** ZoeDepth, Metric Depth  
🔮 **Speed Optimization:** Quantization, distillation  
🔮 **Generalization:** Video editing (frame-consistent)  
🔮 **User Control:** Interactive UI điều chỉnh threshold

---

## Tài liệu tham khảo

- Parihar et al., ICCV 2025
- Rombach et al., CVPR 2022  
- Xia et al., arXiv 2023 (AnyDoor)
- MiDaS, ZoeDepth
