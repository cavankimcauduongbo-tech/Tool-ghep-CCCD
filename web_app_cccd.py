import streamlit as st
from PIL import Image
import cv2
import numpy as np
from rembg import remove, new_session
import io

# --- CẤU HÌNH ---
st.set_page_config(page_title="Tool Ghép CCCD Pro - Kim ATP", page_icon="🆔", layout="centered")

# --- 1. CORE LOGIC (THUẬT TOÁN MỚI) ---

@st.cache_resource
def load_ai_session():
    return new_session("u2net")

def pixel_from_mm(mm, dpi=300):
    return int(mm * dpi / 25.4)

def crop_and_straighten(image_pil, session):
    """
    Thuật toán V2: Tách nền AI -> Tự động xoay thẳng -> Cắt theo khung
    Giữ nguyên góc bo tròn đẹp mắt, không làm méo chữ.
    """
    # 1. Convert PIL to OpenCV
    img_np = np.array(image_pil)
    
    # 2. Dùng AI tách nền (Lấy ảnh PNG trong suốt)
    try:
        # Xóa nền
        no_bg = remove(img_np, session=session)
        
        # Tách kênh Alpha để tìm vật thể
        alpha = no_bg[:, :, 3]
        
        # Tìm contour lớn nhất (là cái thẻ)
        cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return image_pil # Không tìm thấy gì thì trả về ảnh gốc
            
        c = max(cnts, key=cv2.contourArea)
        
        # 3. Tính góc nghiêng để xoay cho thẳng
        rect = cv2.minAreaRect(c)
        (center, (w, h), angle) = rect
        
        # Chuẩn hóa góc xoay
        if w < h:
            angle = angle - 90
            
        # Xoay ảnh
        (h_img, w_img) = no_bg.shape[:2]
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(no_bg, M, (w_img, h_img), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        
        # 4. Cắt (Crop) lại sau khi xoay
        # Tìm lại contour trên ảnh đã xoay để cắt sát lề
        alpha_rotated = rotated[:, :, 3]
        cnts_rot, _ = cv2.findContours(alpha_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_rot:
            c_rot = max(cnts_rot, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c_rot)
            
            # Thêm chút lề (padding) cho thoáng, tránh cắt phạm chữ
            pad = 10
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(w_img - x, w + 2*pad)
            h = min(h_img - y, h + 2*pad)
            
            cropped = rotated[y:y+h, x:x+w]
            
            # Convert về PIL
            return Image.fromarray(cropped)
            
    except Exception as e:
        st.error(f"Lỗi xử lý ảnh: {e}")
        return image_pil

    return Image.fromarray(no_bg)

# --- 2. GIAO DIỆN WEB ---

def main():
    st.markdown("<h1 style='text-align: center; color: #d35400;'>🆔 TOOL GHÉP CCCD PRO v2</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Công nghệ: AI Tách nền + Tự động xoay thẳng + Giữ nguyên góc bo tròn</p>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Cài đặt")
    use_ai = st.sidebar.checkbox("Bật AI Tách nền & Căn chỉnh", value=True)
    
    session = None
    if use_ai:
        with st.spinner("Đang khởi động AI..."):
            session = load_ai_session()

    # Upload
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Mặt Trước")
        front_file = st.file_uploader("Tải ảnh mặt trước", type=['jpg', 'png', 'jpeg'], key="front")
    
    with col2:
        st.subheader("2. Mặt Sau")
        back_file = st.file_uploader("Tải ảnh mặt sau", type=['jpg', 'png', 'jpeg'], key="back")

    if front_file and back_file:
        if st.button("🚀 XỬ LÝ VÀ GHÉP ẢNH", type="primary", use_container_width=True):
            try:
                with st.spinner("Đang tách nền và căn chỉnh..."):
                    # Load ảnh
                    f_img = Image.open(front_file)
                    b_img = Image.open(back_file)

                    # Xử lý
                    if use_ai:
                        img1 = crop_and_straighten(f_img, session)
                        img2 = crop_and_straighten(b_img, session)
                    else:
                        img1 = f_img
                        img2 = b_img

                    # Thông số A4 & Thẻ (Scale chuẩn)
                    DPI = 300
                    # Tăng kích thước thẻ lên xíu (88mm) để bù trừ in ấn cho đẹp
                    CARD_W_MM, CARD_H_MM = 85.6, 53.98
                    
                    A4_W_PX = pixel_from_mm(210, DPI)
                    A4_H_PX = pixel_from_mm(297, DPI)
                    C_W_PX = pixel_from_mm(CARD_W_MM, DPI)
                    C_H_PX = pixel_from_mm(CARD_H_MM, DPI)

                    # Resize ảnh về kích thước chuẩn ID-1
                    # Dùng LANCZOS để giữ nét chữ
                    img1 = img1.resize((C_W_PX, C_H_PX), Image.Resampling.LANCZOS)
                    img2 = img2.resize((C_W_PX, C_H_PX), Image.Resampling.LANCZOS)

                    # Tạo nền A4 trắng
                    canvas = Image.new('RGBA', (A4_W_PX, A4_H_PX), (255, 255, 255, 255))
                    
                    # Tọa độ căn giữa
                    cx = A4_W_PX // 2
                    # Khoảng cách giữa 2 thẻ (khoảng 3cm = 350px) cho thoáng
                    gap = 350 
                    start_y = (A4_H_PX - (C_H_PX * 2 + gap)) // 2 

                    # Dán ảnh (Dùng mask để giữ độ trong suốt của góc bo tròn)
                    canvas.paste(img1, (cx - C_W_PX // 2, start_y), img1)
                    canvas.paste(img2, (cx - C_W_PX // 2, start_y + C_H_PX + gap), img2)

                    # Chuyển sang RGB để lưu PDF
                    final_pdf = canvas.convert('RGB')

                    # Hiển thị
                    st.success("Xong! Ảnh đã được căn thẳng hàng.")
                    st.image(final_pdf, caption="Demo kết quả", use_container_width=True)

                    # Download
                    pdf_buffer = io.BytesIO()
                    final_pdf.save(pdf_buffer, "PDF", resolution=300.0)
                    
                    st.download_button(
                        label="📥 TẢI FILE PDF (Bản đẹp)",
                        data=pdf_buffer.getvalue(),
                        file_name="CCCD_Ghep_KimATP_v2.pdf",
                        mime="application/pdf",
                        type="primary"
                    )

            except Exception as e:
                st.error(f"Có lỗi: {e}")

    # Footer
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey;'>App created by Cà Văn Kim - ATP</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()