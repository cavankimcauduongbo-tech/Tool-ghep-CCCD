import streamlit as st
from PIL import Image
import cv2
import numpy as np
from rembg import remove, new_session
import io
import gc

# --- CẤU HÌNH ---
st.set_page_config(page_title="Tool Ghép CCCD V5.2 (Fix Full)", page_icon="🆔", layout="centered")

# --- 1. CORE LOGIC ---

@st.cache_resource
def load_ai_session():
    # Vẫn dùng bản nhẹ để không sập server
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

def smart_scan_v5_2(image_pil, session):
    # 1. Chuẩn hóa ảnh đầu vào (Fix lỗi bad transparency)
    # Bắt buộc chuyển về RGB để tránh lỗi kênh Alpha
    image_pil = image_pil.convert("RGB")
    
    # Resize nếu ảnh quá lớn để tiết kiệm RAM
    max_size = 1500
    w, h = image_pil.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        image_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)

    img_np = np.array(image_pil)
    
    try:
        # 2. Lấy Mask để tìm vị trí (KHÔNG DÙNG ĐỂ CẮT TRỰC TIẾP)
        # Chỉ lấy cái khuôn hình dáng
        mask_pil = remove(image_pil, session=session, only_mask=True)
        mask = np.array(mask_pil)
        
        # 3. Tìm Contour
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Nếu không tìm thấy hoặc mask lỗi, trả về ảnh gốc đã resize
        if not cnts: return image_pil
        
        c = max(cnts, key=cv2.contourArea)
        
        # 4. Tìm hộp bao quanh (MinAreaRect)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        
        # --- FIX LỖI NUMPY: int0 -> astype(int) ---
        box = box.astype(int)
        
        # 5. Ép phẳng (Perspective Transform) trên ẢNH GỐC
        # (Chiến thuật: Lấy tọa độ từ AI, nhưng cắt trên ảnh màu gốc để không bị mất chữ)
        
        dst_w, dst_h = 1011, 638 # Chuẩn pixel scan ID-1
        rect_pts = order_points(box)
        
        # Logic tự động xoay ngang nếu AI nhận diện nhầm chiều dọc
        # Tính khoảng cách cạnh
        width_check = np.linalg.norm(rect_pts[0] - rect_pts[1])
        height_check = np.linalg.norm(rect_pts[0] - rect_pts[3])
        
        if height_check > width_check:
            # Nếu ảnh đang đứng dọc, xoay điểm lại cho nằm ngang
            rect_pts = np.array([rect_pts[3], rect_pts[0], rect_pts[1], rect_pts[2]], dtype="float32")

        dst_pts = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
        
        # Cắt từ ảnh gốc -> Đảm bảo nét căng, không bị lẹm, không bị bóng ma
        warped = cv2.warpPerspective(img_np, M, (dst_w, dst_h), flags=cv2.INTER_LANCZOS4)
        
        return Image.fromarray(warped)

    except Exception as e:
        # Nếu lỗi quá thì trả về ảnh gốc chứ không để sập web
        st.warning(f"AI gặp khó khăn: {e}. Đang dùng ảnh gốc.")
        return image_pil

# --- 2. GIAO DIỆN WEB ---

def main():
    st.markdown("<h1 style='text-align: center; color: #2980b9;'>🆔 TOOL V5.2 (FINAL FIX)</h1>", unsafe_allow_html=True)
    st.caption("Fix lỗi NumPy & Lỗi mất nét chữ")
    
    use_ai = st.checkbox("Bật chế độ Scan Phẳng", value=True)
    
    session = None
    if use_ai:
        with st.spinner("Đang khởi động AI..."):
            session = load_ai_session()

    col1, col2 = st.columns(2)
    with col1: f_file = st.file_uploader("Mặt Trước", type=['jpg','png','jpeg'], key="f")
    with col2: b_file = st.file_uploader("Mặt Sau", type=['jpg','png','jpeg'], key="b")

    if f_file and b_file:
        if st.button("🚀 XỬ LÝ NGAY", type="primary", use_container_width=True):
            try:
                gc.collect()
                
                with st.spinner("Đang xử lý..."):
                    img1 = Image.open(f_file)
                    img2 = Image.open(b_file)

                    if use_ai:
                        scan1 = smart_scan_v5_2(img1, session)
                        scan2 = smart_scan_v5_2(img2, session)
                    else:
                        scan1, scan2 = img1, img2

                    # Ghép A4
                    A4_W, A4_H = 2480, 3508
                    target_w, target_h = 1011, 638 # Kích thước thẻ chuẩn trên A4
                    
                    scan1 = scan1.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    scan2 = scan2.resize((target_w, target_h), Image.Resampling.LANCZOS)

                    canvas = Image.new('RGB', (A4_W, A4_H), 'white')
                    cx = A4_W // 2
                    gap = 300
                    sy = (A4_H - (target_h * 2 + gap)) // 2 

                    canvas.paste(scan1, (cx - target_w // 2, sy))
                    canvas.paste(scan2, (cx - target_w // 2, sy + target_h + gap))

                    st.success("Thành công!")
                    st.image(canvas, caption="Kết quả V5.2", use_container_width=True)

                    pdf_buffer = io.BytesIO()
                    canvas.save(pdf_buffer, "PDF", resolution=300.0)
                    
                    st.download_button("📥 TẢI PDF", pdf_buffer.getvalue(), "CCCD_Final.pdf", "application/pdf", type="primary")
                    
                    del scan1, scan2, canvas, img1, img2
                    gc.collect()

            except Exception as e:
                st.error(f"Lỗi không mong muốn: {e}")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey;'>App created by Cà Văn Kim - ATP</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()