import streamlit as st
from PIL import Image
import cv2
import numpy as np
from rembg import remove, new_session
import io
import gc
from streamlit_drawable_canvas import st_canvas # Thư viện vẽ lên ảnh

# --- CẤU HÌNH ---
st.set_page_config(page_title="Tool CCCD V16 (Click & Cut)", page_icon="🆔", layout="wide")

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
    """Cắt ảnh từ 4 điểm người dùng click"""
    img_np = np.array(image_pil.convert("RGB"))
    
    # Lấy 4 điểm từ input
    pts = np.array(points, dtype="float32")
    rect_pts = order_points(pts)

    # Kích thước chuẩn ID-1 (300 DPI)
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
    
    # Load ảnh
    image = Image.open(uploaded_file)
    w, h = image.size
    
    # Resize ảnh hiển thị cho vừa màn hình (Canvas không nên quá to)
    display_width = 600
    ratio = display_width / w
    display_height = int(h * ratio)
    
    st.markdown(f"### 🖱️ {label}: Click chuột vào 4 góc thẻ")
    st.caption("Thứ tự click không quan trọng. Click sai thì bấm nút 'Thùng rác' trên thanh công cụ ảnh để xóa làm lại.")

    # TẠO CANVAS ĐỂ CLICK
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Màu vùng chọn
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=image,
        update_streamlit=True,
        height=display_height,
        width=display_width,
        drawing_mode="point", # Chế độ chấm điểm
        point_display_radius=5,
        key=f"canvas_{key_prefix}",
    )

    # Xử lý kết quả click
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        points = [obj for obj in objects if obj["type"] == "circle"]
        
        if len(points) == 4:
            st.success("✅ Đã chọn đủ 4 góc!")
            
            # Quy đổi tọa độ từ Canvas (ảnh nhỏ) về Ảnh gốc (ảnh to)
            real_points = []
            for p in points:
                real_x = p["left"] / ratio
                real_y = p["top"] / ratio
                real_points.append([real_x, real_y])
            
            # Nút xác nhận cắt
            if st.button(f"✂️ CẮT {label.upper()}", key=f"btn_crop_{key_prefix}", type="primary"):
                return warp_from_points(image, real_points)
        elif len(points) > 4:
            st.warning("⚠️ Bạn click quá 4 điểm. Hãy xóa bớt (nút thùng rác).")
        elif len(points) > 0:
            st.info(f"Đã chọn {len(points)}/4 điểm...")
            
    return None

def main():
    st.markdown("<h1 style='text-align: center; color: #d35400;'>🆔 TOOL V16 (CLICK & CUT)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Tương tác trực tiếp trên ảnh - Chuẩn từng pixel</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        f_file = st.file_uploader("Mặt Trước", type=['jpg','png','jpeg'], key="f_up")
    with col2:
        b_file = st.file_uploader("Mặt Sau", type=['jpg','png','jpeg'], key="b_up")

    # Xử lý
    img1_final = None
    img2_final = None

    c1, c2 = st.columns(2)
    
    with c1:
        if f_file:
            cropped_1 = interactive_crop_ui("Mặt Trước", "front", f_file)
            if cropped_1:
                st.session_state['crop_1'] = cropped_1
            
            # Hiện kết quả đã lưu
            if 'crop_1' in st.session_state:
                st.image(st.session_state['crop_1'], caption="Đã cắt xong", width=300)
                img1_final = st.session_state['crop_1']

    with c2:
        if b_file:
            cropped_2 = interactive_crop_ui("Mặt Sau", "back", b_file)
            if cropped_2:
                st.session_state['crop_2'] = cropped_2
            
            if 'crop_2' in st.session_state:
                st.image(st.session_state['crop_2'], caption="Đã cắt xong", width=300)
                img2_final = st.session_state['crop_2']

    # Ghép PDF
    if img1_final and img2_final:
        st.markdown("---")
        if st.button("📄 XUẤT FILE PDF", type="primary", use_container_width=True):
            # Quy chuẩn kích thước
            TARGET_W, TARGET_H = 1011, 638
            scan1 = img1_final.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            scan2 = img2_final.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)

            # Canvas A4
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
            st.download_button("📥 TẢI PDF", pdf_buffer.getvalue(), "CCCD_Interactive.pdf", "application/pdf", type="primary")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey;'>App created by Cà Văn Kim - ATP</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()