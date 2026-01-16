import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from rembg import remove, new_session
import io
import gc

# --- CẤU HÌNH ---
st.set_page_config(page_title="Tool CCCD V8 (Chỉnh Nghiêng)", page_icon="🆔", layout="centered")

# --- CORE LOGIC ---

@st.cache_resource
def load_ai_session():
    return new_session("u2netp")

def enhance_image(image_pil):
    """Tăng tương phản để AI dễ tách nền bàn gỗ hơn"""
    enhancer = ImageEnhance.Contrast(image_pil)
    image_pil = enhancer.enhance(1.5) # Tăng 50% tương phản
    enhancer_sharp = ImageEnhance.Sharpness(image_pil)
    image_pil = enhancer_sharp.enhance(2.0) # Tăng độ nét
    return image_pil

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

def rotate_image(image, angle):
    """Xoay ảnh thủ công để sửa nghiêng"""
    if angle == 0: return image
    # Dùng Bicubic để giữ nét khi xoay
    return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255,255,255))

def smart_scan_v8(image_pil, session, shave_w=15, shave_h=5):
    # 1. Chuẩn hóa & Tăng tương phản đầu vào
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

    # Tạo bản copy đã tăng tương phản để đưa vào AI (giúp tách nền tốt hơn)
    img_for_ai = enhance_image(image_pil_resized)
    img_np_resized = np.array(img_for_ai)
    
    try:
        # 2. Lấy Mask
        mask_pil = remove(img_for_ai, session=session, only_mask=True)
        mask = np.array(mask_pil)
        
        # 3. Tìm Contour
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return image_pil
        c = max(cnts, key=cv2.contourArea)
        
        # 4. THUẬT TOÁN MỚI: ApproxPolyDP (Bắt đa giác 4 cạnh)
        # Cách này chuẩn hơn minAreaRect khi gặp viền bo tròn hoặc bóng
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            # Nếu tìm được đúng 4 góc -> Quá tuyệt
            box = approx.reshape(4, 2)
        else:
            # Nếu không (do bóng làm méo), quay về cách cũ
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            
        box = box.astype(int)
        
        # 5. Quy đổi về ảnh gốc
        if scale_ratio != 1.0:
            box = (box / scale_ratio).astype(int)
            img_np_final = np.array(image_pil)
        else:
            img_np_final = np.array(image_pil)

        # 6. Ép phẳng
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
        
        # Cắt gọt viền
        h_warped, w_warped = warped.shape[:2]
        if w_warped > 2*shave_w and h_warped > 2*shave_h:
            warped_shaved = warped[shave_h:h_warped-shave_h, shave_w:w_warped-shave_w]
            return Image.fromarray(warped_shaved)
        else:
            return Image.fromarray(warped)

    except Exception as e:
        st.warning(f"Lỗi AI: {e}")
        return image_pil

# --- GIAO DIỆN WEB ---

def main():
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🆔 TOOL V8 (CHỈNH NGHIÊNG)</h1>", unsafe_allow_html=True)
    st.caption("AI + Chỉnh tay thủ công cho ca khó")
    
    # --- KHU VỰC ĐIỀU KHIỂN ---
    st.markdown("### 🛠️ Bộ điều khiển")
    
    c1, c2 = st.columns(2)
    with c1:
        shave_w = st.slider("Gọt viền Trái/Phải (px)", 0, 40, 20)
    with c2:
        shave_h = st.slider("Gọt viền Trên/Dưới (px)", 0, 30, 5)
        
    st.markdown("---")
    st.markdown("**🔄 Chỉnh nghiêng thủ công (Nếu AI bị lệch):**")
    r1, r2 = st.columns(2)
    with r1:
        rot_f = st.slider("Xoay Mặt Trước (Độ)", -10.0, 10.0, 0.0, 0.5)
    with r2:
        rot_b = st.slider("Xoay Mặt Sau (Độ)", -10.0, 10.0, 0.0, 0.5)
    
    use_ai = st.checkbox("Bật AI Scan", value=True)
    
    session = None
    if use_ai:
        with st.spinner("Đang tải AI..."):
            session = load_ai_session()

    # Upload
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
                        # Scan bằng AI
                        scan1 = smart_scan_v8(img1, session, shave_w=shave_w, shave_h=shave_h)
                        scan2 = smart_scan_v8(img2, session, shave_w=shave_w, shave_h=shave_h)
                    else:
                        scan1, scan2 = img1, img2
                    
                    # --- BƯỚC MỚI: ÁP DỤNG XOAY THỦ CÔNG ---
                    if rot_f != 0: scan1 = rotate_image(scan1, rot_f)
                    if rot_b != 0: scan2 = rotate_image(scan2, rot_b)

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
                    st.image(canvas, caption=f"Đã gọt {shave_w}px | Xoay {rot_f}° / {rot_b}°", use_container_width=True)

                    pdf_buffer = io.BytesIO()
                    canvas.save(pdf_buffer, "PDF", resolution=300.0)
                    
                    st.download_button("📥 TẢI PDF", pdf_buffer.getvalue(), "CCCD_V8_Pro.pdf", "application/pdf", type="primary")
                    
                    del scan1, scan2, canvas, img1, img2
                    gc.collect()

            except Exception as e:
                st.error(f"Lỗi: {e}")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey;'>App created by Cà Văn Kim - ATP</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()