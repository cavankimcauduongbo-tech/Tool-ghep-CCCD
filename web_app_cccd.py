import streamlit as st
from PIL import Image
import cv2
import numpy as np
from rembg import remove, new_session
import io
import gc

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Tool Ghép CCCD - Kim ATP (Fixed)",
    page_icon="🆔",
    layout="centered"
)

# --- 1. LOGIC XỬ LÝ ẢNH ---

@st.cache_resource
def load_ai_session():
    """Load model AI bản nhẹ để tránh sập web"""
    return new_session("u2netp") # Đã đổi sang u2netp cho mượt

def pixel_from_mm(mm, dpi=300):
    return int(mm * dpi / 25.4)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Tính toán kích thước chuẩn
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)

def process_card_image(uploaded_file, use_ai, session):
    # Đọc ảnh từ file upload
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, 1)
    
    # Resize ảnh nếu quá lớn để tránh tràn RAM (Fix lỗi Over capacity)
    h, w = img_cv.shape[:2]
    if max(h, w) > 1500:
        scale = 1500 / max(h, w)
        img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)))

    orig = img_cv.copy()
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(img_rgb)

    if use_ai:
        try:
            # Lấy mask
            mask_pil = remove(image_pil, session=session, only_mask=True)
            mask = np.array(mask_pil)
            
            # Tìm contour
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                # Lấy contour lớn nhất
                c = max(cnts, key=cv2.contourArea)
                
                # --- SỬA LỖI NGHIÊNG Ở ĐÂY ---
                # Thay vì dùng approxPolyDP (dễ bị méo do bóng), ta dùng minAreaRect
                # minAreaRect sẽ vẽ một hộp chữ nhật bao quanh -> Luôn thẳng góc 90 độ
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = box.astype(int) # Fix lỗi Numpy int0
                
                # Cắt và ép phẳng
                warped = four_point_transform(orig, box)
                
                # Xóa nền lần 2 cho sạch viền đen (Clean cut)
                warped_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
                clean_card = remove(warped_pil, session=session)
                return clean_card

        except Exception as e:
            st.error(f"Lỗi AI: {e}")
    
    return Image.fromarray(img_rgb)

# --- 2. GIAO DIỆN WEB ---

def main():
    st.markdown("<h1 style='text-align: center; color: #0078D7;'>🆔 TOOL GHÉP CCCD (FIX TILT)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Đã sửa lỗi ảnh bị nghiêng & Tối ưu bộ nhớ</p>", unsafe_allow_html=True)
    
    st.sidebar.header("Cài đặt")
    use_ai = st.sidebar.checkbox("Sử dụng AI Tách nền", value=True)
    
    session = None
    if use_ai:
        with st.spinner("Đang khởi động AI..."):
            session = load_ai_session()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Mặt Trước")
        front_file = st.file_uploader("Chọn ảnh mặt trước", type=['jpg', 'png', 'jpeg'], key="front")
    
    with col2:
        st.subheader("2. Mặt Sau")
        back_file = st.file_uploader("Chọn ảnh mặt sau", type=['jpg', 'png', 'jpeg'], key="back")

    if front_file and back_file:
        if st.button("🚀 BẮT ĐẦU XỬ LÝ NGAY", type="primary", use_container_width=True):
            try:
                gc.collect()
                with st.spinner("Đang xử lý ảnh..."):
                    img1 = process_card_image(front_file, use_ai, session)
                    img2 = process_card_image(back_file, use_ai, session)

                    # Thông số A4
                    DPI = 300
                    A4_W, A4_H = pixel_from_mm(210, DPI), pixel_from_mm(297, DPI)
                    # Kích thước chuẩn ID-1
                    C_W, C_H = pixel_from_mm(85.6, DPI), pixel_from_mm(53.98, DPI)

                    img1 = img1.resize((C_W, C_H), Image.Resampling.LANCZOS)
                    img2 = img2.resize((C_W, C_H), Image.Resampling.LANCZOS)

                    canvas = Image.new('RGB', (A4_W, A4_H), 'white')
                    cx = A4_W // 2
                    sy = (A4_H - (C_H * 2 + 150)) // 2 

                    canvas.paste(img1, (cx - C_W // 2, sy))
                    canvas.paste(img2, (cx - C_W // 2, sy + C_H + 150))

                    st.success("Đã xử lý xong!")
                    st.image(canvas, caption="Kết quả (Đã căn thẳng)", use_container_width=True)

                    pdf_buffer = io.BytesIO()
                    canvas.save(pdf_buffer, "PDF", resolution=300.0)
                    
                    st.download_button(
                        label="📥 TẢI VỀ FILE PDF",
                        data=pdf_buffer.getvalue(),
                        file_name="CCCD_Ghep_KimATP.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
                    
                gc.collect()

            except Exception as e:
                st.error(f"Có lỗi xảy ra: {str(e)}")
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: white; background-color: #052c52; padding: 10px; border-radius: 5px;'><b>App created by Cà Văn Kim - ATP</b></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()