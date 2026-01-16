import streamlit as st
from PIL import Image, ImageDraw
import cv2
import numpy as np
from rembg import remove, new_session
import io
import gc
from streamlit_image_coordinates import streamlit_image_coordinates

# --- CẤU HÌNH ---
st.set_page_config(page_title="Tool CCCD V18 (Click Coordinates)", page_icon="🆔", layout="wide")

# --- CORE LOGIC ---

@st.cache_resource
def load_ai_session():
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

def warp_from_points(image_pil, points):
    img_np = np.array(image_pil.convert("RGB"))
    pts = np.array(points, dtype="float32")
    rect_pts = order_points(pts)

    dst_w, dst_h = 1011, 638
    
    dst_pts = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1]], dtype="float32")
    
    # Kiểm tra chiều dọc/ngang
    w_rect = np.linalg.norm(rect_pts[0] - rect_pts[1])
    h_rect = np.linalg.norm(rect_pts[0] - rect_pts[3])
    
    if h_rect > w_rect:
        rect_pts = np.array([rect_pts[3], rect_pts[0], rect_pts[1], rect_pts[2]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    warped = cv2.warpPerspective(img_np, M, (dst_w, dst_h), flags=cv2.INTER_LANCZOS4)
    return Image.fromarray(warped)

# --- UI COMPONENT ---

def interactive_crop_ui(label, key_prefix, uploaded_file):
    if not uploaded_file: return None
    
    # Session state để lưu các điểm đã click
    pts_key = f"{key_prefix}_points"
    if pts_key not in st.session_state:
        st.session_state[pts_key] = []

    # Load ảnh gốc
    image = Image.open(uploaded_file).convert("RGB")
    w_orig, h_orig = image.size
    
    # Resize ảnh hiển thị cho vừa màn hình (khoảng 600px width)
    display_width = 600
    ratio = display_width / w_orig
    display_height = int(h_orig * ratio)
    img_resized = image.resize((display_width, display_height))
    
    st.markdown(f"### 🖱️ {label}")
    st.caption("Click lần lượt vào 4 góc của thẻ. Nếu sai bấm 'Xóa làm lại'.")

    # Vẽ các điểm đã click lên ảnh hiển thị
    img_draw = img_resized.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Vẽ các điểm đã chọn
    points = st.session_state[pts_key]
    for i, p in enumerate(points):
        # p đang là tọa độ thật, cần quy đổi về tọa độ hiển thị
        px = int(p[0] * ratio)
        py = int(p[1] * ratio)
        
        # Vẽ chấm tròn
        r = 8
        color = "#FF0000" if i < 3 else "#00FF00" # Điểm cuối màu xanh
        draw.ellipse((px-r, py-r, px+r, py+r), fill=color, outline="white", width=2)
        draw.text((px+r, py), str(i+1), fill="yellow")

    # --- THÀNH PHẦN CLICK (Thay thế st_canvas) ---
    # Component này chỉ trả về tọa độ click cuối cùng
    value = streamlit_image_coordinates(
        img_draw,
        key=f"coord_{key_prefix}",
        width=display_width,
    )

    # Xử lý sự kiện click
    if value is not None:
        # Lấy tọa độ click trên ảnh hiển thị
        click_x = value["x"]
        click_y = value["y"]
        
        # Quy đổi về tọa độ ảnh gốc
        real_x = click_x / ratio
        real_y = click_y / ratio
        
        # Kiểm tra xem điểm này đã có chưa (tránh click đúp)
        new_point = (real_x, real_y)
        
        # Logic thêm điểm (chỉ thêm nếu chưa đủ 4)
        if len(points) < 4:
            # Kiểm tra trùng lặp đơn giản (nếu click quá gần điểm cũ thì bỏ qua)
            is_duplicate = False
            if len(points) > 0:
                last_pt = points[-1]
                if abs(last_pt[0] - real_x) < 5 and abs(last_pt[1] - real_y) < 5:
                    is_duplicate = True
            
            if not is_duplicate:
                points.append(new_point)
                st.session_state[pts_key] = points
                st.rerun() # Load lại trang để vẽ điểm mới lên ảnh

    # Các nút điều khiển
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("🗑️ Xóa làm lại", key=f"reset_{key_prefix}"):
            st.session_state[pts_key] = []
            st.rerun()
            
    with c2:
        if len(points) == 4:
            if st.button(f"✂️ CẮT {label.upper()} NGAY", key=f"crop_{key_prefix}", type="primary"):
                return warp_from_points(image, points)
        elif len(points) > 0:
            st.info(f"Đã chọn {len(points)}/4 điểm...")

    return None

def main():
    st.markdown("<h1 style='text-align: center; color: #d35400;'>🆔 TOOL V18 (CLICK SIÊU NHẸ)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Fix lỗi màn hình đen - Click chính xác 100%</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1: f_file = st.file_uploader("Mặt Trước", type=['jpg','png','jpeg'], key="f_up")
    with col2: b_file = st.file_uploader("Mặt Sau", type=['jpg','png','jpeg'], key="b_up")

    # Xử lý
    img1_final = None
    img2_final = None

    c1, c2 = st.columns(2)
    
    with c1:
        if f_file:
            cropped_1 = interactive_crop_ui("Mặt Trước", "front", f_file)
            if cropped_1: st.session_state['crop_1_v18'] = cropped_1
            
            if 'crop_1_v18' in st.session_state:
                st.image(st.session_state['crop_1_v18'], caption="Kết quả Mặt Trước", width=350)
                img1_final = st.session_state['crop_1_v18']

    with c2:
        if b_file:
            cropped_2 = interactive_crop_ui("Mặt Sau", "back", b_file)
            if cropped_2: st.session_state['crop_2_v18'] = cropped_2
            
            if 'crop_2_v18' in st.session_state:
                st.image(st.session_state['crop_2_v18'], caption="Kết quả Mặt Sau", width=350)
                img2_final = st.session_state['crop_2_v18']

    # Ghép PDF
    if img1_final and img2_final:
        st.markdown("---")
        if st.button("📄 XUẤT FILE PDF A4", type="primary", use_container_width=True):
            TARGET_W, TARGET_H = 1011, 638
            scan1 = img1_final.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            scan2 = img2_final.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)

            A4_W, A4_H = pixel_from_mm(210, 300), pixel_from_mm(297, 300)
            canvas = Image.new('RGB', (A4_W, A4_H), 'white')
            
            cx = A4_W // 2
            gap = 350
            sy = (A4_H - (TARGET_H * 2 + gap)) // 2 

            canvas.paste(scan1, (cx - TARGET_W // 2, sy))
            canvas.paste(scan2, (cx - TARGET_W // 2, sy + TARGET_H + gap))

            pdf_buffer = io.BytesIO()
            canvas.save(pdf_buffer, "PDF", resolution=300.0)
            
            st.success("Xong!")
            st.download_button("📥 TẢI PDF VỀ MÁY", pdf_buffer.getvalue(), "CCCD_Click_V18.pdf", "application/pdf", type="primary")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey;'>App created by Cà Văn Kim - ATP</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()