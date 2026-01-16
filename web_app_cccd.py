import streamlit as st
from PIL import Image
import cv2
import numpy as np
from rembg import remove, new_session
import io
import gc
import base64
from streamlit_drawable_canvas import st_canvas 

# --- CẤU HÌNH ---
st.set_page_config(page_title="Tool CCCD V17 (Fix Mất Ảnh)", page_icon="🆔", layout="wide")

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
    
    w_rect = np.linalg.norm(rect_pts[0] - rect_pts[1])
    h_rect = np.linalg.norm(rect_pts[0] - rect_pts[3])
    
    if h_rect > w_rect:
        rect_pts = np.array([rect_pts[3], rect_pts[0], rect_pts[1], rect_pts[2]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    warped = cv2.warpPerspective(img_np, M, (dst_w, dst_h), flags=cv2.INTER_LANCZOS4)
    return Image.fromarray(warped)

# --- HÀM CỨU HỘ: Chuyển ảnh sang Base64 để né lỗi mất hình ---
def get_image_base64(img_pil):
    buff = io.BytesIO()
    img_pil.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode()
    return img_str

# --- UI COMPONENT ---

def interactive_crop_ui(label, key_prefix, uploaded_file):
    if not uploaded_file: return None
    
    # Load ảnh
    image = Image.open(uploaded_file)
    w, h = image.size
    
    # Tính toán kích thước hiển thị
    display_width = 650
    ratio = display_width / w
    display_height = int(h * ratio)
    
    # Resize ảnh hiển thị (để load cho nhanh)
    img_resized = image.resize((display_width, display_height))
    
    st.markdown(f"### 🖱️ {label}")
    st.info("👇 Click chuột vào 4 góc của thẻ, sau đó bấm nút Cắt bên dưới.")

    # --- KHU VỰC CANVAS (Vẽ lên ảnh) ---
    # Chuyển ảnh thành nền Canvas (Dùng Base64 để fix lỗi không hiện)
    bg_base64 = get_image_base64(img_resized)
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=Image.open(io.BytesIO(base64.b64decode(bg_base64))), # Trick để load ảnh an toàn
        update_streamlit=True,
        height=display_height,
        width=display_width,
        drawing_mode="point", # Chế độ chấm điểm
        point_display_radius=6,
        key=f"canvas_{key_prefix}",
    )

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        points = [obj for obj in objects if obj["type"] == "circle"]
        
        # Nút hành động
        if len(points) == 4:
            st.success("✅ Đã chọn đủ 4 điểm!")
            if st.button(f"✂️ CẮT {label.upper()} NGAY", key=f"btn_crop_{key_prefix}", type="primary"):
                # Quy đổi tọa độ
                real_points = []
                for p in points:
                    real_x = p["left"] / ratio
                    real_y = p["top"] / ratio
                    real_points.append([real_x, real_y])
                return warp_from_points(image, real_points)
        elif len(points) > 4:
            st.warning("⚠️ Bạn chấm thừa điểm rồi! Bấm vào nút 'Thùng rác' trên thanh công cụ ảnh để xóa làm lại.")
        elif len(points) > 0:
            st.warning(f"Đang chọn: {len(points)}/4 điểm...")
            
    return None

def main():
    st.markdown("<h1 style='text-align: center; color: #d35400;'>🆔 TOOL V17 (FIX MẤT ẢNH)</h1>", unsafe_allow_html=True)

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
            if cropped_1: st.session_state['crop_1'] = cropped_1
            
            if 'crop_1' in st.session_state:
                st.image(st.session_state['crop_1'], caption="Kết quả Mặt Trước", width=350)
                img1_final = st.session_state['crop_1']

    with c2:
        if b_file:
            cropped_2 = interactive_crop_ui("Mặt Sau", "back", b_file)
            if cropped_2: st.session_state['crop_2'] = cropped_2
            
            if 'crop_2' in st.session_state:
                st.image(st.session_state['crop_2'], caption="Kết quả Mặt Sau", width=350)
                img2_final = st.session_state['crop_2']

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
            st.download_button("📥 TẢI PDF VỀ MÁY", pdf_buffer.getvalue(), "CCCD_Interactive.pdf", "application/pdf", type="primary")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey;'>App created by Cà Văn Kim - ATP</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()