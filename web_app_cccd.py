import streamlit as st
from PIL import Image
import cv2
import numpy as np
from rembg import remove, new_session
import io
import gc

# --- CẤU HÌNH ---
st.set_page_config(
    page_title="Tool CCCD V11 (Chuẩn Scan)",
    page_icon="🆔",
    layout="centered"
)

# --- CORE LOGIC ---

@st.cache_resource
def load_ai_session():
    # Dùng u2netp (bản nhẹ) để xử lý nhanh và không sập web
    return new_session("u2netp")

def pixel_from_mm(mm, dpi=300):
    return int(mm * dpi / 25.4)

def order_points(pts):
    """Sắp xếp 4 điểm: Trên-Trái, Trên-Phải, Dưới-Phải, Dưới-Trái"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def crop_center_ratio(img, target_ratio=1.585):
    """
    Cắt xén bớt phần thừa (bóng/nền) để ảnh đạt đúng tỷ lệ thẻ CCCD
    target_ratio = 85.6 / 53.98 ≈ 1.585
    """
    h, w = img.shape[:2]
    current_ratio = w / h
    
    if current_ratio > target_ratio:
        # Ảnh đang bị dài quá (thừa 2 bên) -> Cắt bớt chiều ngang
        new_w = int(h * target_ratio)
        offset = (w - new_w) // 2
        return img[:, offset:offset+new_w]
    elif current_ratio < target_ratio:
        # Ảnh đang bị cao quá (thừa trên dưới - do bóng) -> Cắt bớt chiều dọc
        new_h = int(w / target_ratio)
        offset = (h - new_h) // 2
        return img[offset:offset+new_h, :]
    else:
        return img

def smart_process_v11(image_pil, session):
    # 1. Chuẩn hóa đầu vào
    image_pil = image_pil.convert("RGB")
    
    # Resize ảnh quá khổ để tiết kiệm RAM
    max_dim = 1500
    w, h = image_pil.size
    scale = 1.0
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image_pil = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    img_np = np.array(image_pil)
    
    try:
        # 2. Lấy Mask để tìm vị trí
        # Chỉ lấy mask đen trắng, không cắt vội
        mask_pil = remove(image_pil, session=session, only_mask=True)
        mask = np.array(mask_pil)
        
        # 3. Tìm Contour
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return image_pil
        c = max(cnts, key=cv2.contourArea)
        
        # 4. QUAN TRỌNG: Dùng minAreaRect để lấy HỘP CHỮ NHẬT
        # Hàm này luôn trả về hình chữ nhật vuông vắn, không bao giờ bị méo
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        
        # 5. Sắp xếp điểm và Ép phẳng (Perspective Transform)
        rect_pts = order_points(box)
        
        # Tính chiều rộng/cao của hộp bao tìm được
        w_box = np.linalg.norm(rect_pts[0] - rect_pts[1])
        h_box = np.linalg.norm(rect_pts[0] - rect_pts[3])
        
        # Logic tự động xoay ngang nếu thẻ đang đứng dọc
        if h_box > w_box:
            # Xoay thứ tự điểm
            rect_pts = np.array([rect_pts[3], rect_pts[0], rect_pts[1], rect_pts[2]], dtype="float32")
            w_box, h_box = h_box, w_box # Hoán đổi kích thước
            
        # Kích thước đích tạm thời (giữ nguyên độ phân giải gốc của vùng cắt)
        dst_w = int(w_box)
        dst_h = int(h_box)
        
        dst_pts = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
        
        # Cắt từ ảnh gốc (img_np) -> Kết quả là 1 hình chữ nhật, nhưng có thể bị sai tỷ lệ do bóng
        warped = cv2.warpPerspective(img_np, M, (dst_w, dst_h), flags=cv2.INTER_LANCZOS4)
        
        # 6. BƯỚC QUYẾT ĐỊNH: Cắt gọt về tỷ lệ chuẩn (1.585)
        # Bước này sẽ loại bỏ phần bóng thừa làm sai kích thước
        final_img_np = crop_center_ratio(warped, target_ratio=1.5858)
        
        # Trả về ảnh sạch (không cần xóa nền lần 2 để tránh mất góc)
        return Image.fromarray(final_img_np)

    except Exception as e:
        st.error(f"Lỗi xử lý: {e}")
        return image_pil

# --- GIAO DIỆN WEB ---

def main():
    st.markdown("<h1 style='text-align: center; color: #27ae60;'>🆔 TOOL V11 (CHUẨN SCAN)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Thẳng tắp - Vuông vức - Đúng tỷ lệ</p>", unsafe_allow_html=True)
    
    use_ai = st.sidebar.checkbox("Bật chế độ Auto Scan", value=True)
    
    session = None
    if use_ai:
        with st.spinner("Đang khởi động AI..."):
            session = load_ai_session()

    col1, col2 = st.columns(2)
    with col1: f_file = st.file_uploader("Mặt Trước", type=['jpg','png','jpeg'], key="f")
    with col2: b_file = st.file_uploader("Mặt Sau", type=['jpg','png','jpeg'], key="b")

    if f_file and b_file:
        if st.button("🚀 SCAN VÀ GHÉP ẢNH", type="primary", use_container_width=True):
            try:
                gc.collect()
                with st.spinner("Đang scan phẳng và căn chỉnh..."):
                    img1 = Image.open(f_file)
                    img2 = Image.open(b_file)

                    if use_ai:
                        # Scan và ép phẳng
                        scan1 = smart_process_v11(img1, session)
                        scan2 = smart_process_v11(img2, session)
                    else:
                        scan1, scan2 = img1, img2

                    # --- GIAI ĐOẠN GHÉP A4 ---
                    # 1. Quy đổi kích thước chuẩn 300 DPI
                    # Thẻ CCCD: 85.6mm x 53.98mm -> pixel
                    TARGET_W = pixel_from_mm(85.6, 300) # ~1011 px
                    TARGET_H = pixel_from_mm(53.98, 300) # ~638 px
                    
                    # 2. Resize ảnh đã scan về đúng kích thước pixel này
                    # Vì ở bước trên đã crop_center_ratio nên resize sẽ không bị méo
                    scan1 = scan1.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
                    scan2 = scan2.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)

                    # 3. Tạo khổ A4
                    A4_W, A4_H = pixel_from_mm(210, 300), pixel_from_mm(297, 300)
                    canvas = Image.new('RGB', (A4_W, A4_H), 'white')
                    
                    # 4. Căn giữa
                    cx = A4_W // 2
                    gap = 350 # Khoảng cách giữa 2 mặt
                    start_y = (A4_H - (TARGET_H * 2 + gap)) // 2 

                    # Dán ảnh (có thêm viền đen mảnh 1px cho giống scan - tùy chọn)
                    # Ở đây mình dán trơn cho đẹp
                    canvas.paste(scan1, (cx - TARGET_W // 2, start_y))
                    canvas.paste(scan2, (cx - TARGET_W // 2, start_y + TARGET_H + gap))

                    st.success("Hoàn thành!")
                    st.image(canvas, caption="Kết quả chuẩn Scan (V11)", use_container_width=True)

                    pdf_buffer = io.BytesIO()
                    canvas.save(pdf_buffer, "PDF", resolution=300.0)
                    
                    st.download_button("📥 TẢI PDF", pdf_buffer.getvalue(), "CCCD_V11_Scan.pdf", "application/pdf", type="primary")
                    
                gc.collect()

            except Exception as e:
                st.error(f"Lỗi: {e}")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey;'>App created by Cà Văn Kim - ATP</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()