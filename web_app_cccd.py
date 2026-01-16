import streamlit as st
from PIL import Image
import cv2
import numpy as np
from rembg import remove, new_session
import io
import gc

# --- CẤU HÌNH ---
st.set_page_config(page_title="Tool Ghép CCCD V5.1 (Fix)", page_icon="🆔", layout="centered")

# --- 1. CORE LOGIC ---

@st.cache_resource
def load_ai_session():
    # Dùng u2netp (bản nhẹ) để tiết kiệm RAM
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

def smart_scan_v5(image_pil, session):
    # 1. Resize ảnh đầu vào nếu quá lớn (Giảm tải RAM)
    max_size = 1500
    w, h = image_pil.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        image_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)

    img_np = np.array(image_pil)
    
    try:
        # 2. Lấy Mask (Dùng model nhẹ)
        mask_pil = remove(image_pil, session=session, only_mask=True)
        mask = np.array(mask_pil)
        
        # 3. Bào mòn (Erosion) để chống nghiêng
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        
        # 4. Tìm Contour
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return image_pil
        
        c = max(cnts, key=cv2.contourArea)
        
        # 5. MinAreaRect (Tìm hộp bao)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        
        # --- FIX LỖI NUMPY Ở ĐÂY ---
        # Thay vì dùng np.int0(box) (đã bị xóa), ta dùng .astype(int)
        box = box.astype(int) 
        # ---------------------------
        
        # 6. Ép phẳng (Perspective Transform)
        dst_w, dst_h = 1011, 638
        rect_pts = order_points(box)
        dst_pts = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
        
        # Cắt từ ảnh gốc
        warped = cv2.warpPerspective(img_np, M, (dst_w, dst_h), flags=cv2.INTER_LANCZOS4)
        
        # Xóa nền lần cuối
        warped_pil = Image.fromarray(warped)
        final_clean = remove(warped_pil, session=session) 
        
        return final_clean

    except Exception as e:
        st.error(f"Lỗi xử lý: {e}")
        return image_pil

# --- 2. GIAO DIỆN WEB ---

def main():
    st.markdown("<h1 style='text-align: center; color: #d35400;'>🆔 TOOL V5.1 (STABLE)</h1>", unsafe_allow_html=True)
    st.caption("Đã sửa lỗi NumPy & Tối ưu bộ nhớ")
    
    use_ai = st.checkbox("Bật AI (Chế độ Lite)", value=True)
    
    session = None
    if use_ai:
        with st.spinner("Đang tải AI Lite..."):
            session = load_ai_session()

    col1, col2 = st.columns(2)
    with col1: f_file = st.file_uploader("Mặt Trước", type=['jpg','png','jpeg'], key="f")
    with col2: b_file = st.file_uploader("Mặt Sau", type=['jpg','png','jpeg'], key="b")

    if f_file and b_file:
        if st.button("🚀 XỬ LÝ NGAY", type="primary", use_container_width=True):
            try:
                gc.collect() # Dọn rác bộ nhớ
                
                with st.spinner("Đang xử lý..."):
                    img1 = Image.open(f_file)
                    img2 = Image.open(b_file)

                    if use_ai:
                        scan1 = smart_scan_v5(img1, session)
                        scan2 = smart_scan_v5(img2, session)
                    else:
                        scan1, scan2 = img1, img2

                    # Ghép A4
                    A4_W, A4_H = 2480, 3508
                    target_w, target_h = 1011, 638
                    
                    scan1 = scan1.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    scan2 = scan2.resize((target_w, target_h), Image.Resampling.LANCZOS)

                    canvas = Image.new('RGB', (A4_W, A4_H), 'white')
                    cx = A4_W // 2
                    gap = 300
                    sy = (A4_H - (target_h * 2 + gap)) // 2 

                    canvas.paste(scan1, (cx - target_w // 2, sy), scan1)
                    canvas.paste(scan2, (cx - target_w // 2, sy + target_h + gap), scan2)

                    st.success("Thành công!")
                    st.image(canvas, caption="Kết quả V5.1", use_container_width=True)

                    pdf_buffer = io.BytesIO()
                    canvas.save(pdf_buffer, "PDF", resolution=300.0)
                    
                    st.download_button("📥 TẢI PDF", pdf_buffer.getvalue(), "CCCD_V5_Fix.pdf", "application/pdf", type="primary")
                    
                    del scan1, scan2, canvas, img1, img2
                    gc.collect()

            except Exception as e:
                st.error(f"Lỗi: {e}. Hãy thử ảnh nhẹ hơn.")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey;'>App created by Cà Văn Kim - ATP</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()