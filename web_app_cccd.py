import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from rembg import remove, new_session
import io
import gc

# --- CẤU HÌNH ---
st.set_page_config(page_title="Tool CCCD V13 (Auto Pro)", page_icon="🆔", layout="centered")

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

def get_extreme_points(mask):
    """
    Tìm 4 điểm cực trị của contour thay vì tìm hộp bao (minAreaRect).
    Giúp loại bỏ các phần bóng lồi ra ở cạnh bên.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    
    # Tìm 4 điểm cực: Trái nhất, Phải nhất, Trên cùng, Dưới cùng
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # Gom lại thành 4 điểm góc giả định
    # Lưu ý: Đây là cách xấp xỉ hình thoi, cần biến đổi về hình chữ nhật
    # Nên dùng approxPolyDP sẽ tốt hơn cho hình chữ nhật cứng
    
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True) # Tăng hệ số lên 0.04 để bắt góc cứng hơn
    
    if len(approx) == 4:
        return approx.reshape(4, 2)
    else:
        # Nếu không ra 4 góc, quay về minAreaRect nhưng thu nhỏ box lại 5%
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        return box

def smart_auto_v13(image_pil, session):
    # 1. Tăng cường ảnh đầu vào để AI dễ nhìn
    image_pil = image_pil.convert("RGB")
    enhancer = ImageEnhance.Contrast(image_pil)
    image_pil = enhancer.enhance(1.5) # Tăng tương phản
    
    # Resize
    max_dim = 1500
    w, h = image_pil.size
    scale = 1.0
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image_pil = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    img_np = np.array(image_pil)
    
    try:
        # 2. Lấy Mask
        mask_pil = remove(image_pil, session=session, only_mask=True)
        mask = np.array(mask_pil)
        
        # 3. KỸ THUẬT MỚI: Morphological Close
        # Lấp đầy các lỗ hổng bên trong thẻ (nếu có) và làm mượt viền
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 4. Tìm điểm góc thông minh
        box = get_extreme_points(mask)
        if box is None: return image_pil
        
        box = box.astype(int)
        
        # 5. Ép phẳng chuẩn tỷ lệ
        rect_pts = order_points(box)
        
        # Tự động xoay ngang
        w_box = np.linalg.norm(rect_pts[0] - rect_pts[1])
        h_box = np.linalg.norm(rect_pts[0] - rect_pts[3])
        
        if h_box > w_box:
            rect_pts = np.array([rect_pts[3], rect_pts[0], rect_pts[1], rect_pts[2]], dtype="float32")
        
        # Kích thước đích chuẩn ID-1 (Pixel 300dpi)
        dst_w = 1011
        dst_h = 638
        
        dst_pts = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
        warped = cv2.warpPerspective(img_np, M, (dst_w, dst_h), flags=cv2.INTER_LANCZOS4)
        
        # 6. Gọt nhẹ viền (Auto-Shave) 10px để loại bỏ mép thừa
        # Vì đã ép đúng kích thước 1011x638 nên gọt 10px là an toàn
        shave = 10
        warped_clean = warped[shave:dst_h-shave, shave:dst_w-shave]
        
        return Image.fromarray(warped_clean)

    except Exception as e:
        st.warning(f"Lỗi xử lý: {e}. Dùng ảnh gốc.")
        return image_pil

# --- GIAO DIỆN WEB ---

def main():
    st.markdown("<h1 style='text-align: center; color: #2980b9;'>🆔 TOOL V13 (AUTO PRO)</h1>", unsafe_allow_html=True)
    st.caption("Tự động hoàn toàn - Không cần chỉnh tay")
    
    use_ai = st.checkbox("Bật chế độ Auto", value=True)
    session = None
    if use_ai:
        with st.spinner("Đang khởi động hệ thống..."):
            session = load_ai_session()

    col1, col2 = st.columns(2)
    with col1: f_file = st.file_uploader("Mặt Trước", type=['jpg','png','jpeg'], key="f")
    with col2: b_file = st.file_uploader("Mặt Sau", type=['jpg','png','jpeg'], key="b")

    if f_file and b_file:
        if st.button("🚀 XỬ LÝ TỰ ĐỘNG", type="primary", use_container_width=True):
            try:
                gc.collect()
                with st.spinner("Đang phân tích và cắt gọt..."):
                    img1 = Image.open(f_file)
                    img2 = Image.open(b_file)

                    if use_ai:
                        scan1 = smart_auto_v13(img1, session)
                        scan2 = smart_auto_v13(img2, session)
                    else:
                        scan1, scan2 = img1, img2

                    # --- GHÉP A4 ---
                    # Quy chuẩn kích thước sau khi gọt
                    # Ban đầu 1011x638 -> Gọt 10px mỗi bên -> Còn 991x618
                    # Cần resize lại về 1011x638 để in ra đúng kích thước thật
                    
                    TARGET_W = pixel_from_mm(85.6, 300)
                    TARGET_H = pixel_from_mm(53.98, 300)
                    
                    scan1 = scan1.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
                    scan2 = scan2.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)

                    A4_W, A4_H = pixel_from_mm(210, 300), pixel_from_mm(297, 300)
                    canvas = Image.new('RGB', (A4_W, A4_H), 'white')
                    
                    cx = A4_W // 2
                    gap = 350
                    sy = (A4_H - (TARGET_H * 2 + gap)) // 2 

                    canvas.paste(scan1, (cx - TARGET_W // 2, sy))
                    canvas.paste(scan2, (cx - TARGET_W // 2, sy + TARGET_H + gap))

                    st.success("Xong! Ảnh đã được nắn thẳng tự động.")
                    st.image(canvas, caption="Kết quả V13 Auto", use_container_width=True)

                    pdf_buffer = io.BytesIO()
                    canvas.save(pdf_buffer, "PDF", resolution=300.0)
                    
                    st.download_button("📥 TẢI PDF", pdf_buffer.getvalue(), "CCCD_Auto_V13.pdf", "application/pdf", type="primary")
                    
                gc.collect()

            except Exception as e:
                st.error(f"Lỗi: {e}")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey;'>App created by Cà Văn Kim - ATP</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()