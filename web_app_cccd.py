import streamlit as st
from PIL import Image
import cv2
import numpy as np
from rembg import remove, new_session
import io
import gc

# --- CẤU HÌNH ---
st.set_page_config(page_title="Tool Ghép CCCD V6 (Razor Cut)", page_icon="🆔", layout="centered")

# --- 1. CORE LOGIC ---

@st.cache_resource
def load_ai_session():
    # Vẫn dùng bản nhẹ u2netp
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

def smart_scan_v6(image_pil, session, shave_pixels=8):
    """
    V6: Thuật toán V5.2 + Bước gọt viền thủ công (Razor Cut)
    shave_pixels: Số pixel cắt lẹm vào trong để loại bỏ viền thừa.
    """
    # 1. Chuẩn hóa ảnh đầu vào
    image_pil = image_pil.convert("RGB")
    
    # Resize để tiết kiệm RAM
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
        # 2. Lấy Mask để tìm vị trí
        mask_pil = remove(image_pil_resized, session=session, only_mask=True)
        mask = np.array(mask_pil)
        
        # 3. Tìm Contour
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return image_pil
        c = max(cnts, key=cv2.contourArea)
        
        # 4. Tìm hộp bao quanh (MinAreaRect)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = box.astype(int) # Fix lỗi NumPy
        
        # 5. Quy đổi tọa độ hộp về ảnh gốc (full resolution)
        if scale_ratio != 1.0:
            box = (box / scale_ratio).astype(int)
            img_np_final = np.array(image_pil) # Dùng ảnh gốc độ phân giải cao
        else:
            img_np_final = img_np_resized

        # 6. Ép phẳng (Perspective Transform)
        # Tăng kích thước đích lên một chút để bù cho việc gọt sau này
        dst_w_raw, dst_h_raw = 1030, 650 
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
        # BƯỚC MỚI TRONG V6: RAZOR CUT (Gọt viền)
        # ==================================================
        h_warped, w_warped = warped.shape[:2]
        # Cắt sâu vào trong 'shave_pixels' ở mỗi cạnh
        if w_warped > 2*shave_pixels and h_warped > 2*shave_pixels:
            warped_shaved = warped[shave_pixels:h_warped-shave_pixels, shave_pixels:w_warped-shave_pixels]
            return Image.fromarray(warped_shaved)
        else:
            # Nếu ảnh quá nhỏ không gọt được thì trả về ảnh gốc
            return Image.fromarray(warped)

    except Exception as e:
        st.warning(f"AI gặp khó khăn: {e}. Đang dùng ảnh gốc.")
        return image_pil

# --- 2. GIAO DIỆN WEB ---

def main():
    st.markdown("<h1 style='text-align: center; color: #e74c3c;'>🆔 TOOL V6 (CẮT SÁT VIỀN)</h1>", unsafe_allow_html=True)
    st.caption("Chế độ Razor Cut: Loại bỏ hoàn toàn viền thừa")
    
    # Thanh trượt điều chỉnh độ cắt sát
    shave_amount = st.slider("🪒 Độ cắt sát viền (Pixel)", min_value=0, max_value=20, value=8, help="Tăng lên để cắt sâu hơn vào trong thẻ.")
    
    use_ai = st.checkbox("Bật chế độ Scan & Cắt sát", value=True)
    
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
                
                with st.spinner(f"Đang xử lý (Cắt sát {shave_amount}px)..."):
                    img1 = Image.open(f_file)
                    img2 = Image.open(b_file)

                    if use_ai:
                        # Truyền thêm tham số shave_pixels
                        scan1 = smart_scan_v6(img1, session, shave_pixels=shave_amount)
                        scan2 = smart_scan_v6(img2, session, shave_pixels=shave_amount)
                    else:
                        scan1, scan2 = img1, img2

                    # Ghép A4
                    A4_W, A4_H = 2480, 3508
                    target_w, target_h = 1011, 638 # Kích thước chuẩn cuối cùng
                    
                    # Resize ảnh đã gọt về đúng chuẩn ID-1
                    scan1 = scan1.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    scan2 = scan2.resize((target_w, target_h), Image.Resampling.LANCZOS)

                    canvas = Image.new('RGB', (A4_W, A4_H), 'white')
                    cx = A4_W // 2
                    gap = 300
                    sy = (A4_H - (target_h * 2 + gap)) // 2 

                    canvas.paste(scan1, (cx - target_w // 2, sy))
                    canvas.paste(scan2, (cx - target_w // 2, sy + target_h + gap))

                    st.success("Thành công!")
                    st.image(canvas, caption=f"Kết quả V6 (Đã gọt {shave_amount}px)", use_container_width=True)

                    pdf_buffer = io.BytesIO()
                    canvas.save(pdf_buffer, "PDF", resolution=300.0)
                    
                    st.download_button("📥 TẢI PDF", pdf_buffer.getvalue(), "CCCD_V6_Razor.pdf", "application/pdf", type="primary")
                    
                    del scan1, scan2, canvas, img1, img2
                    gc.collect()

            except Exception as e:
                st.error(f"Lỗi không mong muốn: {e}")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey;'>App created by Cà Văn Kim - ATP</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()