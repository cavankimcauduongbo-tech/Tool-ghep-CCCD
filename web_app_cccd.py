import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import cv2
import numpy as np
from rembg import remove, new_session
import io
import gc

# --- CẤU HÌNH ---
st.set_page_config(page_title="Tool CCCD V14 (Bắt Cạnh Cứng)", page_icon="🆔", layout="centered")

# --- CORE LOGIC ---

@st.cache_resource
def load_ai_session():
    return new_session("u2netp")

def pixel_from_mm(mm, dpi=300):
    return int(mm * dpi / 25.4)

def order_points(pts):
    """Sắp xếp 4 điểm: TL, TR, BR, BL"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_real_edges(image_pil, ai_mask_cv):
    """
    V14 MAGIC: Dùng Canny Edge để tìm cạnh thẻ thật sự bên trong vùng AI tìm thấy.
    Giúp loại bỏ bóng mờ mà AI hay bị nhầm.
    """
    # 1. Convert ảnh sang grayscale
    gray = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
    
    # 2. Làm mờ nhẹ để loại bỏ nhiễu hạt (bụi)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Dùng Canny để tìm cạnh sắc nét (Thẻ nhựa có cạnh rất nét)
    # Threshold 30/150 giúp bắt các đường viền cứng
    edged = cv2.Canny(blurred, 30, 150)
    
    # 4. Chỉ quan tâm đến các cạnh nằm TRONG vùng AI đã tìm thấy (Mask)
    # Để tránh bắt nhầm vân gỗ hay vật thể khác xa cái thẻ
    # Dãn vùng mask ra một chút để đảm bảo bao trọn thẻ
    kernel = np.ones((15,15), np.uint8)
    dilated_mask = cv2.dilate(ai_mask_cv, kernel, iterations=2)
    
    # Kết hợp: Cạnh sắc nét AND Vùng AI
    combined = cv2.bitwise_and(edged, edged, mask=dilated_mask)
    
    return combined

def smart_process_v14(image_pil, session, scale_factor=0.0):
    """
    scale_factor: % thu/phóng khung hình (-0.05 là thu nhỏ 5%, 0.05 là phóng to)
    """
    # 1. Chuẩn hóa
    image_pil = image_pil.convert("RGB")
    
    # Resize an toàn
    max_dim = 1500
    w_orig, h_orig = image_pil.size
    resize_scale = 1.0
    if max(w_orig, h_orig) > max_dim:
        resize_scale = max_dim / max(w_orig, h_orig)
        new_w, new_h = int(w_orig * resize_scale), int(h_orig * resize_scale)
        image_pil = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    img_np = np.array(image_pil)
    
    try:
        # 2. Bước 1: AI tìm vùng thô (Sơ bộ)
        mask_pil = remove(image_pil, session=session, only_mask=True)
        mask = np.array(mask_pil)
        
        # 3. Bước 2: Tìm cạnh cứng (Tinh chỉnh) - Bỏ qua bóng
        edges = get_real_edges(image_pil, mask)
        
        # 4. Tìm contour trên các cạnh cứng đó
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not cnts: 
            # Nếu không tìm thấy cạnh cứng (ảnh quá mờ), quay về dùng mask của AI
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not cnts: return image_pil

        # Lấy contour lớn nhất (là cái thẻ)
        c = max(cnts, key=cv2.contourArea)
        
        # 5. Vẽ hộp bao quanh (MinAreaRect) -> Đảm bảo vuông vức
        rect = cv2.minAreaRect(c)
        (center, (w_box, h_box), angle) = rect
        
        # --- XỬ LÝ SCALE (THU/PHÓNG) TỪ SLIDER ---
        # Nếu người dùng muốn phóng to/thu nhỏ khung cắt
        if scale_factor != 0.0:
            w_box = w_box * (1 + scale_factor)
            h_box = h_box * (1 + scale_factor)
            rect = (center, (w_box, h_box), angle)
        # ------------------------------------------

        box = cv2.boxPoints(rect)
        box = box.astype(int)
        
        # 6. Ép phẳng (Perspective Transform)
        rect_pts = order_points(box)
        
        # Tính kích thước hộp
        w_rect = np.linalg.norm(rect_pts[0] - rect_pts[1])
        h_rect = np.linalg.norm(rect_pts[0] - rect_pts[3])
        
        # Logic xoay ngang
        if h_rect > w_rect:
            rect_pts = np.array([rect_pts[3], rect_pts[0], rect_pts[1], rect_pts[2]], dtype="float32")
            w_rect, h_rect = h_rect, w_rect
            
        # Kích thước đích chuẩn ID-1 (Pixel 300dpi)
        dst_w = 1011
        dst_h = 638
        
        dst_pts = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
        
        # Warp từ ảnh gốc (img_np)
        # borderValue=255 để nếu cắt lẹm ra ngoài thì điền màu trắng
        warped = cv2.warpPerspective(img_np, M, (dst_w, dst_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        return Image.fromarray(warped)

    except Exception as e:
        st.warning(f"Lỗi: {e}")
        return image_pil

# --- GIAO DIỆN WEB ---

def main():
    st.markdown("<h1 style='text-align: center; color: #e67e22;'>🆔 TOOL V14 (BẮT CẠNH CỨNG)</h1>", unsafe_allow_html=True)
    st.caption("Công nghệ dò cạnh nhựa để loại bỏ bóng mờ")
    
    # --- THANH ĐIỀU KHIỂN TINH CHỈNH ---
    st.markdown("### 🎛️ Tinh chỉnh khung cắt")
    st.info("Nếu ảnh bị cắt lẹm mất chữ -> Kéo sang Phải (+). Nếu ảnh bị thừa viền đen -> Kéo sang Trái (-)")
    
    # Slider cho phép thu phóng khung cắt từ -10% đến +10%
    scale_percent = st.slider("Thu/Phóng khung cắt (%)", -10, 10, 0, 1)
    scale_factor = scale_percent / 100.0
    
    use_ai = st.checkbox("Bật chế độ Auto Scan", value=True)
    session = None
    if use_ai:
        with st.spinner("Đang tải AI..."):
            session = load_ai_session()

    col1, col2 = st.columns(2)
    with col1: f_file = st.file_uploader("Mặt Trước", type=['jpg','png','jpeg'], key="f")
    with col2: b_file = st.file_uploader("Mặt Sau", type=['jpg','png','jpeg'], key="b")

    if f_file and b_file:
        if st.button("🚀 SCAN VÀ GHÉP ẢNH", type="primary", use_container_width=True):
            try:
                gc.collect()
                with st.spinner(f"Đang xử lý (Zoom {scale_percent}%)..."):
                    img1 = Image.open(f_file)
                    img2 = Image.open(b_file)

                    if use_ai:
                        # Truyền tham số scale vào hàm xử lý
                        scan1 = smart_process_v14(img1, session, scale_factor)
                        scan2 = smart_process_v14(img2, session, scale_factor)
                    else:
                        scan1, scan2 = img1, img2

                    # --- GHÉP A4 ---
                    TARGET_W = pixel_from_mm(85.6, 300)
                    TARGET_H = pixel_from_mm(53.98, 300)
                    
                    # Resize về chuẩn in ấn
                    scan1 = scan1.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
                    scan2 = scan2.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)

                    A4_W, A4_H = pixel_from_mm(210, 300), pixel_from_mm(297, 300)
                    canvas = Image.new('RGB', (A4_W, A4_H), 'white')
                    
                    cx = A4_W // 2
                    gap = 350
                    sy = (A4_H - (TARGET_H * 2 + gap)) // 2 

                    canvas.paste(scan1, (cx - TARGET_W // 2, sy))
                    canvas.paste(scan2, (cx - TARGET_W // 2, sy + TARGET_H + gap))

                    st.success("Xong!")
                    st.image(canvas, caption=f"Kết quả (Zoom {scale_percent}%)", use_container_width=True)

                    pdf_buffer = io.BytesIO()
                    canvas.save(pdf_buffer, "PDF", resolution=300.0)
                    
                    st.download_button("📥 TẢI PDF", pdf_buffer.getvalue(), "CCCD_V14.pdf", "application/pdf", type="primary")
                    
                gc.collect()

            except Exception as e:
                st.error(f"Lỗi: {e}")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey;'>App created by Cà Văn Kim - ATP</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()