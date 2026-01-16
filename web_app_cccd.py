import streamlit as st
from PIL import Image
import cv2
import numpy as np
from rembg import remove, new_session
import io
import gc

# --- CẤU HÌNH ---
st.set_page_config(page_title="Tool CCCD V7 (Cắt Độc Lập)", page_icon="🆔", layout="centered")

# --- CORE LOGIC ---

@st.cache_resource
def load_ai_session():
    return new_session("u2netp") # Bản nhẹ

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

def smart_scan_v7(image_pil, session, shave_w=15, shave_h=5):
    """
    V7: Cắt độc lập chiều ngang (w) và dọc (h)
    """
    # 1. Chuẩn hóa
    image_pil = image_pil.convert("RGB")
    
    # Resize
    max_size = 1500
    w_orig, h_orig = image_pil.size
    scale_ratio = 1.0
    if max(w_orig, h_orig) > max_size:
        scale_ratio = max_size / max(w_orig, h_orig)
        new_size = (int(w_orig * scale_ratio), int(h_orig * scale_ratio))
        image_pil_resized = image_pil.resize(new_size, Image.Resampling.LANCZOS)
    else:
        image_pil_resized = image_pil

    img_np_resized = np.array(image_pil_resized)
    
    try:
        # 2. Lấy Mask
        mask_pil = remove(image_pil_resized, session=session, only_mask=True)
        mask = np.array(mask_pil)
        
        # 3. Tìm Contour
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return image_pil
        c = max(cnts, key=cv2.contourArea)
        
        # 4. Tìm hộp bao quanh
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        
        # 5. Quy đổi về ảnh gốc
        if scale_ratio != 1.0:
            box = (box / scale_ratio).astype(int)
            img_np_final = np.array(image_pil)
        else:
            img_np_final = img_np_resized

        # 6. Ép phẳng (Perspective Transform)
        # Tăng kích thước đệm để gọt
        dst_w_raw, dst_h_raw = 1040, 660 
        rect_pts = order_points(box)
        
        # Logic xoay ngang
        width_check = np.linalg.norm(rect_pts[0] - rect_pts[1])
        height_check = np.linalg.norm(rect_pts[0] - rect_pts[3])
        if height_check > width_check:
             rect_pts = np.array([rect_pts[3], rect_pts[0], rect_pts[1], rect_pts[2]], dtype="float32")

        dst_pts = np.array([[0, 0], [dst_w_raw-1, 0], [dst_w_raw-1, dst_h_raw-1], [0, dst_h_raw-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
        warped = cv2.warpPerspective(img_np_final, M, (dst_w_raw, dst_h_raw), flags=cv2.INTER_LANCZOS4)
        
        # ==================================================
        # V7: CẮT ĐỘC LẬP (Independent Shave)
        # ==================================================
        h_warped, w_warped = warped.shape[:2]
        
        # Kiểm tra điều kiện để không cắt lỗi ảnh
        if w_warped > 2*shave_w and h_warped > 2*shave_h:
            # Cắt trên/dưới theo shave_h, trái/phải theo shave_w
            warped_shaved = warped[shave_h:h_warped-shave_h, shave_w:w_warped-shave_w]
            return Image.fromarray(warped_shaved)
        else:
            return Image.fromarray(warped)

    except Exception as e:
        st.warning(f"Lỗi xử lý: {e}")
        return image_pil

# --- GIAO DIỆN WEB ---

def main():
    st.markdown("<h1 style='text-align: center; color: #16a085;'>🆔 TOOL V7 (CẮT ĐỘC LẬP)</h1>", unsafe_allow_html=True)
    st.caption("Chỉnh dao cắt riêng cho 4 cạnh")
    
    # --- THANH ĐIỀU KHIỂN DAO CẮT ---
    st.markdown("### 🪒 Cấu hình dao cắt")
    c1, c2 = st.columns(2)
    with c1:
        # Mặc định 5px cho trên dưới (để không mất chữ)
        shave_h = st.slider("Cắt Trên/Dưới (px)", 0, 30, 5) 
    with c2:
        # Mặc định 15px cho trái phải (để gọt sạch viền thừa mà bạn đang gặp)
        shave_w = st.slider("Cắt Trái/Phải (px)", 0, 40, 15)
    
    use_ai = st.checkbox("Bật chế độ AI Scan", value=True)
    
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
                with st.spinner("Đang gọt giũa ảnh..."):
                    img1 = Image.open(f_file)
                    img2 = Image.open(b_file)

                    if use_ai:
                        # Truyền 2 thông số cắt riêng biệt
                        scan1 = smart_scan_v7(img1, session, shave_w=shave_w, shave_h=shave_h)
                        scan2 = smart_scan_v7(img2, session, shave_w=shave_w, shave_h=shave_h)
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

                    canvas.paste(scan1, (cx - target_w // 2, sy))
                    canvas.paste(scan2, (cx - target_w // 2, sy + target_h + gap))

                    st.success("Xong!")
                    st.image(canvas, caption=f"Đã cắt: Dọc {shave_h}px | Ngang {shave_w}px", use_container_width=True)

                    pdf_buffer = io.BytesIO()
                    canvas.save(pdf_buffer, "PDF", resolution=300.0)
                    
                    st.download_button("📥 TẢI PDF", pdf_buffer.getvalue(), "CCCD_V7_Clean.pdf", "application/pdf", type="primary")
                    
                    del scan1, scan2, canvas, img1, img2
                    gc.collect()

            except Exception as e:
                st.error(f"Lỗi: {e}")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey;'>App created by Cà Văn Kim - ATP</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()