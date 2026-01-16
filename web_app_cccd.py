import streamlit as st
from PIL import Image
import cv2
import numpy as np
from rembg import remove, new_session
import io

# --- CẤU HÌNH ---
st.set_page_config(page_title="Tool Ghép CCCD Scan V4", page_icon="🆔", layout="centered")

# --- 1. CORE LOGIC (THUẬT TOÁN V4: SCAN BOX) ---

@st.cache_resource
def load_ai_session():
    return new_session("u2net")

def pixel_from_mm(mm, dpi=300):
    return int(mm * dpi / 25.4)

def order_points(pts):
    """Sắp xếp 4 điểm: Trên-Trái, Trên-Phải, Dưới-Phải, Dưới-Trái"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left: tổng (x+y) nhỏ nhất
    # Bottom-right: tổng (x+y) lớn nhất
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right: hiệu (y-x) nhỏ nhất
    # Bottom-left: hiệu (y-x) lớn nhất
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def smart_scan_transform(image_pil, session):
    """
    V4: Sử dụng minAreaRect để tìm khung bao ảo -> Đảm bảo thẳng tuyệt đối
    """
    # 1. Convert sang OpenCV
    img_np = np.array(image_pil)
    orig = img_np.copy()
    
    # 2. AI Tách nền lấy Mask
    try:
        # Chỉ lấy mask (đen trắng) để xử lý cho nhanh và chính xác
        mask_pil = remove(image_pil, session=session, only_mask=True)
        mask = np.array(mask_pil)
        
        # 3. Tìm Contour lớn nhất (Vật thể chính)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return image_pil
        
        c = max(cnts, key=cv2.contourArea)
        
        # 4. MAGIC STEP: Tìm hình chữ nhật bao quanh (Rotated Rectangle)
        # Thay vì tìm góc nhọn (dễ sai do bo góc), ta tìm hình hộp bao quanh
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # 5. Ép phẳng (Warp) dựa trên cái hộp đó
        # Kích thước chuẩn ID-1 (tỉ lệ)
        dst_w = 1011 # pixel chuẩn 300dpi
        dst_h = 638
        
        # Sắp xếp 4 điểm nguồn
        rect_pts = order_points(box)
        
        # 4 điểm đích (Hình chữ nhật thẳng đứng)
        dst_pts = np.array([
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1]], dtype="float32")
        
        # Tính ma trận biến đổi
        M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
        
        # Cắt vật thể từ ảnh gốc (đã tách nền hoặc chưa tùy chọn)
        # Ở đây ta cắt từ ảnh gốc ban đầu (có nền) rồi lát nữa rembg đè lên sau
        # HOẶC cắt từ ảnh đã xóa nền. 
        # Tốt nhất: Cắt từ ảnh gốc -> Xóa nền lại (để viền đẹp hơn) 
        # NHƯNG để tối ưu tốc độ: Xóa nền trước -> Tìm hộp -> Cắt.
        
        # Thực hiện lại bước xóa nền full màu để lấy ảnh kết quả
        no_bg = remove(img_np, session=session) # Ảnh PNG trong suốt
        
        # Warp cái ảnh đã xóa nền
        warped = cv2.warpPerspective(no_bg, M, (dst_w, dst_h), flags=cv2.INTER_LANCZOS4)
        
        return Image.fromarray(warped)

    except Exception as e:
        st.error(f"Lỗi xử lý: {e}")
        return image_pil

# --- 2. GIAO DIỆN WEB ---

def main():
    st.markdown("<h1 style='text-align: center; color: #2ecc71;'>🆔 TOOL GHÉP CCCD SCAN (V4)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Chế độ Scan phẳng - Thẳng tắp tuyệt đối</p>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Cài đặt")
    use_ai = st.sidebar.checkbox("AI Auto Scan", value=True)
    
    session = None
    if use_ai:
        with st.spinner("Đang tải AI Engine..."):
            session = load_ai_session()

    col1, col2 = st.columns(2)
    with col1:
        f_file = st.file_uploader("Mặt Trước", type=['jpg', 'png', 'jpeg'], key="f")
    with col2:
        b_file = st.file_uploader("Mặt Sau", type=['jpg', 'png', 'jpeg'], key="b")

    if f_file and b_file:
        if st.button("🚀 BẮT ĐẦU QUÉT & GHÉP", type="primary", use_container_width=True):
            try:
                with st.spinner("Đang quét ảnh..."):
                    img1 = Image.open(f_file)
                    img2 = Image.open(b_file)

                    if use_ai:
                        # Chạy thuật toán V4
                        scan1 = smart_scan_transform(img1, session)
                        scan2 = smart_scan_transform(img2, session)
                    else:
                        scan1 = img1
                        scan2 = img2

                    # --- GIAI ĐOẠN GHÉP ---
                    # Thông số A4 (300 DPI)
                    A4_W, A4_H = 2480, 3508 
                    
                    # Resize về đúng chuẩn kích thước thật (85.6mm x 54mm)
                    # 1011x638 là kích thước pixel chuẩn scan
                    target_w, target_h = 1011, 638
                    
                    scan1 = scan1.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    scan2 = scan2.resize((target_w, target_h), Image.Resampling.LANCZOS)

                    # Tạo Canvas A4
                    canvas = Image.new('RGB', (A4_W, A4_H), 'white')
                    cx = A4_W // 2
                    gap = 300 # Khoảng cách giữa 2 ảnh
                    sy = (A4_H - (target_h * 2 + gap)) // 2 

                    # Dán (dùng mask của chính nó để giữ độ trong suốt nếu có)
                    # scan1, scan2 đang là mode RGBA (do rembg tạo ra)
                    canvas.paste(scan1, (cx - target_w // 2, sy), scan1)
                    canvas.paste(scan2, (cx - target_w // 2, sy + target_h + gap), scan2)

                    st.success("Đã xử lý xong!")
                    
                    # Preview
                    st.image(canvas, caption="Kết quả Scan", use_container_width=True)

                    # Download
                    pdf_buffer = io.BytesIO()
                    canvas.save(pdf_buffer, "PDF", resolution=300.0)
                    
                    st.download_button(
                        label="📥 TẢI FILE PDF SCAN",
                        data=pdf_buffer.getvalue(),
                        file_name="CCCD_Scan_KimATP.pdf",
                        mime="application/pdf",
                        type="primary"
                    )

            except Exception as e:
                st.error(f"Có lỗi: {e}")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey;'>App created by Cà Văn Kim - ATP</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()