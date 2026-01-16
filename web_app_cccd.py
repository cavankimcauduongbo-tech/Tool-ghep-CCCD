import streamlit as st
from PIL import Image
import cv2
import numpy as np
from rembg import remove, new_session
import io
import gc

# --- CẤU HÌNH ---
st.set_page_config(page_title="Tool CCCD V9 (Chuẩn Tỷ Lệ)", page_icon="🆔", layout="centered")

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

def fixed_ratio_transform(image, pts):
    """
    Ép ảnh về tỷ lệ chuẩn ID Card (85.6mm x 53.98mm)
    Khắc phục hoàn toàn lỗi méo hình.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Tính chiều rộng/cao hiện tại để kiểm tra hướng
    width_current = np.linalg.norm(tr - tl)
    height_current = np.linalg.norm(tr - br)
    
    # Kích thước đích CỐ ĐỊNH (Chuẩn pixel 300dpi)
    # Tương đương 85.6mm x 54mm
    dst_w = 1011
    dst_h = 638
    
    # TỰ ĐỘNG XOAY: Nếu ảnh gốc đang đứng dọc (cao > rộng), xoay điểm lại
    if height_current > width_current:
        # Xoay thứ tự điểm 90 độ để khớp với khung ngang
        rect = np.array([bl, tl, tr, br], dtype="float32")

    # Ma trận đích luôn là hình chữ nhật chuẩn
    dst = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Warp với nền trắng (borderValue=255) để mất viền đen
    warped = cv2.warpPerspective(image, M, (dst_w, dst_h), 
                                 flags=cv2.INTER_LANCZOS4, 
                                 borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=(255, 255, 255))
    return warped

def process_card_v9(uploaded_file, use_ai, session):
    # Đọc ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, 1)
    
    # Resize để tránh sập RAM
    h, w = img_cv.shape[:2]
    if max(h, w) > 1500:
        scale = 1500 / max(h, w)
        img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)))

    orig = img_cv.copy()
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(img_rgb)

    if use_ai:
        try:
            # 1. Lấy Mask để tìm vị trí
            mask_pil = remove(image_pil, session=session, only_mask=True)
            mask = np.array(mask_pil)
            
            # 2. Tìm Contour
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                
                # 3. MinAreaRect (Vẽ hộp bao quanh)
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = box.astype(int)
                
                # 4. Ép phẳng với TỶ LỆ KHÓA CỨNG (Fix méo)
                warped = fixed_ratio_transform(orig, box)
                
                # 5. Xóa nền lần cuối (Clean cut)
                warped_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
                clean_card = remove(warped_pil, session=session)
                
                # Chuyển về nền trắng thay vì trong suốt (để in đẹp)
                background = Image.new("RGB", clean_card.size, (255, 255, 255))
                background.paste(clean_card, mask=clean_card.split()[3])
                
                return background

        except Exception as e:
            st.error(f"Lỗi AI: {e}")
    
    return Image.fromarray(img_rgb)

# --- GIAO DIỆN WEB ---

def main():
    st.markdown("<h1 style='text-align: center; color: #8e44ad;'>🆔 TOOL V9 (CHUẨN TỶ LỆ)</h1>", unsafe_allow_html=True)
    st.caption("Khắc phục hoàn toàn lỗi méo hình & viền đen")
    
    use_ai = st.sidebar.checkbox("Sử dụng AI", value=True)
    session = None
    if use_ai:
        with st.spinner("Đang tải AI..."):
            session = load_ai_session()

    col1, col2 = st.columns(2)
    with col1: f_file = st.file_uploader("Mặt Trước", type=['jpg','png','jpeg'], key="f")
    with col2: b_file = st.file_uploader("Mặt Sau", type=['jpg','png','jpeg'], key="b")

    if f_file and b_file:
        if st.button("🚀 XỬ LÝ NGAY", type="primary", use_container_width=True):
            try:
                gc.collect()
                with st.spinner("Đang nắn chỉnh tỷ lệ..."):
                    img1 = process_card_v9(f_file, use_ai, session)
                    img2 = process_card_v9(b_file, use_ai, session)

                    # Ghép A4
                    A4_W, A4_H = pixel_from_mm(210), pixel_from_mm(297)
                    target_w, target_h = 1011, 638 # Kích thước chuẩn đã fix
                    
                    img1 = img1.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    img2 = img2.resize((target_w, target_h), Image.Resampling.LANCZOS)

                    canvas = Image.new('RGB', (A4_W, A4_H), 'white')
                    cx = A4_W // 2
                    sy = (A4_H - (target_h * 2 + 150)) // 2 

                    canvas.paste(img1, (cx - target_w // 2, sy))
                    canvas.paste(img2, (cx - target_w // 2, sy + target_h + 150))

                    st.success("Thành công!")
                    st.image(canvas, caption="Kết quả V9 (Không méo)", use_container_width=True)

                    pdf_buffer = io.BytesIO()
                    canvas.save(pdf_buffer, "PDF", resolution=300.0)
                    
                    st.download_button("📥 TẢI PDF", pdf_buffer.getvalue(), "CCCD_V9.pdf", "application/pdf", type="primary")
                    
                gc.collect()

            except Exception as e:
                st.error(f"Lỗi: {e}")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey;'>App created by Cà Văn Kim - ATP</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()