import streamlit as st
from PIL import Image
import cv2
import numpy as np
from rembg import remove, new_session
import io

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Tool Ghép CCCD - Kim ATP",
    page_icon="🆔",
    layout="centered"
)

# --- 1. LOGIC XỬ LÝ ẢNH (GIỮ NGUYÊN TỪ BẢN DESKTOP) ---

@st.cache_resource
def load_ai_session():
    """Load model AI một lần duy nhất để web chạy nhanh"""
    return new_session("u2net")

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
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def process_card_image(uploaded_file, use_ai, session):
    # Đọc ảnh từ file upload
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, 1)
    
    orig = img_cv.copy()
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    if use_ai:
        try:
            no_bg_image = remove(img_rgb, session=session)
            alpha_channel = no_bg_image[:, :, 3]
            cnts, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
            
            screenCnt = None
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    screenCnt = approx
                    break
            
            if screenCnt is not None:
                warped = four_point_transform(orig, screenCnt.reshape(4, 2))
                return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            elif len(cnts) > 0:
                x, y, w, h = cv2.boundingRect(cnts[0])
                crop = orig[y:y+h, x:x+w]
                return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        except Exception as e:
            st.error(f"Lỗi AI: {e}")
    
    return Image.fromarray(img_rgb)

# --- 2. GIAO DIỆN WEB (STREAMLIT UI) ---

def main():
    # Tiêu đề
    st.markdown("<h1 style='text-align: center; color: #0078D7;'>🆔 TOOL GHÉP CCCD ONLINE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Tự động tách nền, cắt góc và ghép vào khổ A4</p>", unsafe_allow_html=True)
    
    # Sidebar cấu hình
    st.sidebar.header("Cài đặt")
    use_ai = st.sidebar.checkbox("Sử dụng AI Tách nền", value=True)
    st.sidebar.info("Nếu ảnh chụp đã cắt sẵn, hãy bỏ chọn AI để chạy nhanh hơn.")
    
    # Load AI Session
    session = None
    if use_ai:
        with st.spinner("Đang khởi động AI..."):
            session = load_ai_session()

    # Upload file
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Mặt Trước")
        front_file = st.file_uploader("Chọn ảnh mặt trước", type=['jpg', 'png', 'jpeg'], key="front")
    
    with col2:
        st.subheader("2. Mặt Sau")
        back_file = st.file_uploader("Chọn ảnh mặt sau", type=['jpg', 'png', 'jpeg'], key="back")

    # Nút xử lý
    if front_file and back_file:
        if st.button("🚀 BẮT ĐẦU XỬ LÝ NGAY", type="primary", use_container_width=True):
            try:
                with st.spinner("Đang xử lý ảnh... Vui lòng đợi..."):
                    # Xử lý 2 ảnh
                    img1 = process_card_image(front_file, use_ai, session)
                    img2 = process_card_image(back_file, use_ai, session)

                    # Thông số A4
                    DPI = 300
                    A4_W, A4_H = pixel_from_mm(210, DPI), pixel_from_mm(297, DPI)
                    C_W, C_H = pixel_from_mm(85.6, DPI), pixel_from_mm(53.98, DPI)

                    # Resize
                    img1 = img1.resize((C_W, C_H), Image.Resampling.LANCZOS)
                    img2 = img2.resize((C_W, C_H), Image.Resampling.LANCZOS)

                    # Tạo Canvas
                    canvas = Image.new('RGB', (A4_W, A4_H), 'white')
                    cx = A4_W // 2
                    sy = (A4_H - (C_H * 2 + 150)) // 2 

                    canvas.paste(img1, (cx - C_W // 2, sy))
                    canvas.paste(img2, (cx - C_W // 2, sy + C_H + 150))

                    # Hiển thị kết quả Preview (Resize nhỏ để xem trên web)
                    st.success("Đã xử lý xong!")
                    st.image(canvas, caption="Kết quả xem trước (Ảnh gốc tải về sẽ nét 100%)", use_container_width=True)

                    # Lưu vào bộ nhớ đệm để tải xuống
                    pdf_buffer = io.BytesIO()
                    canvas.save(pdf_buffer, "PDF", resolution=300.0)
                    pdf_bytes = pdf_buffer.getvalue()

                    # Nút tải xuống
                    st.download_button(
                        label="📥 TẢI VỀ FILE PDF (Chuẩn in ấn)",
                        data=pdf_bytes,
                        file_name="CCCD_Ghep_KimATP.pdf",
                        mime="application/pdf",
                        type="primary"
                    )

            except Exception as e:
                st.error(f"Có lỗi xảy ra: {str(e)}")
    
    # --- FOOTER ---
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: white; background-color: #052c52; padding: 10px; border-radius: 5px;'>
            <b>App created by Cà Văn Kim - ATP</b>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()