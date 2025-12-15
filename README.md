# Zero-Shot Depth-Aware Image Editing - Presentation

## 📋 Tổng Quan

Đây là tài liệu thuyết trình chi tiết về **Zero-Shot Depth-Aware Image Editing with Diffusion Models** - một phương pháp tiên tiến để chỉnh sửa ảnh dựa trên hiểu biết 3D mà không cần training lại.

https://openaccess.thecvf.com/content/ICCV2025/papers/Parihar_Zero-Shot_Depth_Aware_Image_Editing_with_Diffusion_Models_ICCV_2025_paper.pdf
link tài liệu chính

## 🎯 Tính Năng Chính

✅ **Fixed Header Navigation** - Thanh điều hướng cố định với active state
✅ **Interactive Table of Contents** - Click vào mục lục để nhảy tới section
✅ **Progress Bar** - Thanh tiến độ scroll theo dõi vị trí hiện tại
✅ **Smooth Scrolling** - Cuộn mượt mà khi chuyển section
✅ **SVG Diagrams** - Sơ đồ minh họa sinh động (không cần hình ảnh ngoài)
✅ **Responsive Design** - Tối ưu cho desktop, tablet, mobile
✅ **Beautiful Styling** - Thiết kế hiện đại với gradient, shadows, animations
✅ **Visual Hierarchy** - Color-coded sections, icons, callout boxes

## 📁 Cấu Trúc Tài Liệu

```
index.html                 ← Tài liệu chính (mở file này)
DOCUMENTATION.md           ← Markdown gốc (tham khảo)
README.md                  ← File này
```

## 🚀 Cách Sử Dụng

### 1. Mở Tài Liệu
```bash
# Mở file index.html bằng trình duyệt
# Windows: Double-click vào index.html
# hoặc: Chuột phải > Open with > Chrome/Edge/Firefox
```

### 2. Điều Hướng
- **Top Navigation**: Click vào menu items (Vấn Đề, Mô Hình, Quy Trình, Kết Quả, Hiểu Biết)
- **Table of Contents**: Lên đầu trang để click vào từng mục
- **Progress Bar**: Xem ở dưới header (đỏ + xanh)
- **Back to Top**: Click nút tròn đỏ ở góc phải để lên đầu

### 3. Nội Dung Các Section

#### 🎯 **Vấn Đề & Giải Pháp**
- Tại sao Diffusion Models không hiểu 3D
- Giải pháp: DeGLaD + FeatGLaC (Zero-Shot)
- So sánh trước/sau

#### 🤖 **Các Mô Hình Chính**
1. **LDM/Stable Diffusion** - Backbone chính của hệ thống
   - Denoising iterative process
   - 50 steps từ noise → hình ảnh
   
2. **Depth Estimation (MiDaS/ZoeDepth)** - Tạo bản đồ độ sâu
   - RGB Image → Depth Map
   - Sáng = gần, Tối = xa
   
3. **AnyDoor** - Bảo vệ hình dáng vật thể
   - Encoding vật thể → Feature Vector
   - Tiêm vào U-Net để preserve identity

#### ⚙️ **Quy Trình Xử Lý**
- **DeGLaD** (Depth-Guided Layer Decomposition)
  - Tách ảnh thành layer dựa Depth Map
  - Tiền cảnh | Vật thể | Hậu cảnh
  
- **FeatGLaC** (Feature Guided Layer Compositing)
  - Trộn layer trong feature space (latent)
  - Inject feature vào mỗi denoising step
  - Kết quả: Composition tự nhiên

#### 📊 **Kết Quả & Big Data**
- **So sánh phương pháp**: Copy-Paste vs Inpainting vs DeGLaD+FeatGLaC
- **Map-Reduce Pattern**: Xử lý 10GB ảnh song song
- **Hiệu suất**: 3.5x speedup với 4 GPUs
- **Tính toán**: 7 giờ sequential → ~2 giờ parallel

#### 💡 **Những Hiểu Biết Chính**
- Zero-Shot = Linh hoạt
- 3D-Aware = Thông minh
- Feature Space = Chất lượng
- Map-Reduce = Mở rộng
- Modular = Nâng cấp
- Identity Preservation = Sáng tạo

#### 📚 **Tài Liệu Tham Khảo**
- Bài báo gốc (ICCV 2025)
- Các công bố nền tảng (Diffusion, Depth, Identity Preservation)
- Links Google Scholar

## 🎨 Thiết Kế

### Màu Sắc
- 🔴 **Đỏ (#e74c3c)**: Problem, Error, Alert
- 🟢 **Xanh (#27ae60)**: Solution, Success
- 🟠 **Cam (#f39c12)**: Warning, Information
- 🔵 **Xanh dương (#3498db)**: Process, Technical
- 🟣 **Tím (#667eea)**: Gradient, Premium

### Icons & Emojis
- ❌ Problem/Issue
- ✅ Solution/Success
- 1️⃣ 2️⃣ 3️⃣ Numbering
- 🎯 Target/Objective
- 🤖 AI/Model
- ⚙️ Process/Technical
- 📊 Data/Analytics
- 💡 Insight/Idea
- 📚 Reference/Learning

## 💻 Kỹ Thuật

### HTML/CSS/JS Stack
- **HTML5**: Semantic markup, SVG diagrams
- **CSS3**: Flexbox, Grid, Gradient, Animation, Responsive
- **Vanilla JS**: Smooth scroll, progress tracking, active nav

### Responsive Breakpoints
- **Desktop**: >1200px (full layout)
- **Tablet**: 768px - 1200px (grid adjustments)
- **Mobile**: <768px (single column, hidden nav)

### Performance
- Lightweight (single HTML file ~150KB)
- No external dependencies
- SVG diagrams (scalable, crisp)
- CSS animations (GPU accelerated)

## 🔧 Tùy Chỉnh

### Thay Đổi Màu Sắc
Tìm section `:root` hoặc CSS variables (nếu có) và update giá trị hex colors.

### Thêm Section Mới
```html
<section id="your-section-id">
    <div class="container">
        <h2>Your Section Title</h2>
        <!-- Content here -->
    </div>
</section>
```

Rồi thêm link vào navigation:
```html
<li><a href="#your-section-id" class="nav-link">Your Title</a></li>
```

### Chỉnh Sửa Diagram
Các SVG diagrams nằm trong `<div class="diagram-container">`. Edit viewBox, coordinates, colors trực tiếp.

## 📱 Browser Support

✅ Chrome/Edge 90+
✅ Firefox 88+
✅ Safari 14+
✅ Mobile browsers (iOS Safari, Chrome Android)

## 📖 Cách Thuyết Trình

1. **Mở file index.html trên projector/screen**
2. **Khởi động bằng Table of Contents** (Slide đầu tiên)
3. **Dẫn dắt audience theo từng section**:
   - Đặt câu hỏi: "Tại sao khó?"
   - Giới thiệu vấn đề (Vấn Đề & Giải Pháp)
   - Giải thích kỹ thuật (Các Mô Hình)
   - Demo pipeline (Quy Trình)
   - Chỉ ra kết quả (Kết Quả)
   - Tổng kết những hiểu biết chính (Insights)
4. **Dùng navigation để quay lại sections trước** nếu cần
5. **Kết thúc bằng Tài Liệu Tham Khảo** (cho những ai muốn đi sâu)

## ✨ Highlights

- **Không cần training**: Zero-Shot approach
- **Hiểu 3D**: Depth-aware composition
- **Chất lượng cao**: Feature space blending
- **Mở rộng được**: Map-Reduce parallelization
- **Sáng tạo**: Identity preservation

## 📞 Ghi Chú

- Tài liệu được tạo ngày: December 8, 2025
- Dựa trên bài báo: "Zero-Shot Depth-Aware Image Editing with Diffusion Models" (ICCV 2025)
- Dùng cho: Thuyết trình học tập, workshop, seminar

## 🎓 Tài Nguyên Học Tập Thêm

- **Diffusion Models**: Ho et al. (NeurIPS 2020), Rombach et al. (CVPR 2022)
- **Depth Estimation**: Bhat et al. - MiDaS (ICCV 2021), ZoeDepth (NeurIPS 2023)
- **Identity Preservation**: Jiang et al. - AnyDoor (CVPR 2023)
- **Google Scholar**: [Tìm kiếm các bài báo liên quan](https://scholar.google.com)

---

**Chúc bạn thuyết trình tốt! 🚀**
