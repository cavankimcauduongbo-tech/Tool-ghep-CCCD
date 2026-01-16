import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance
import cv2
import numpy as np
from rembg import remove, new_session
import io
import gc
from streamlit_image_coordinates import streamlit_image_coordinates

# --- CẤU HÌNH ---
st.set_page_config(page_title="Tool CCCD V19 (Hybrid)", page_icon="🆔", layout="wide")

# --- CORE LOGIC CHUNG ---

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

# --- LOGIC TỰ ĐỘNG (AUTO) ---
def crop_center_ratio(img, target_ratio=1.5858):
    h, w = img.shape[:2]
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        offset = (w - new_w) // 2
        return img[:, offset:offset+new_w]
    elif current_ratio < target_ratio:
        new_h = int(w / target_ratio)
        offset = (h - new_h) // 2
        return img[offset:offset+new_h, :]
    return img

def auto_process_image(image_pil, session):
    # Chuẩn hóa
    image_pil = image_pil.convert("RGB")
    enhancer = ImageEnhance.Contrast(image_pil)
    image_pil = enhancer.enhance(1.5) # Tăng tương phản
    
    img_np = np.array(image_pil)
    
    try:
        # 1. Lấy Mask
        mask_pil = remove(image_pil, session=session, only_mask=True)
        mask = np.array(mask_pil)
        
        # 2. Tìm Contour
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return image_pil
        c = max(cnts, key=cv2.contourArea)
        
        # 3. MinAreaRect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        
        # 4. Warp
        rect_pts = order_points(box)
        w_box = np.linalg.norm(rect_pts[0] - rect_pts[1])
        h_box = np.linalg.norm(rect_pts[0] - rect_pts[3])
        
        dst_w, dst_h = int(w_box), int(h_box)
        
        if h_box > w_box:
            rect_pts = np.array([rect_pts[3], rect_pts[0], rect_pts[1], rect_pts[2]], dtype="float32")
            dst_w, dst_h = dst_h, dst_w # Swap

        dst_pts = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
        warped = cv2.warpPerspective(img_np, M, (dst_w, dst_h), flags=cv2.INTER_LANCZOS4)
        
        # 5. Crop tỷ lệ chuẩn (1.585)
        final_img = crop_center_ratio(warped, 1.5858)
        
        return Image.fromarray(final_img)
    except:
        return image_pil

# --- LOGIC CHỈNH TAY (MANUAL) ---
def warp_from_points(image_pil, points):
    img_np = np.array(image_pil.convert("RGB"))
    pts = np.array(points, dtype="float32")
    rect_pts = order_points(pts)

    dst_w, dst_h = 1011, 638
    dst_pts = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32")
    
    w_src = np.linalg.norm(rect_pts[0] - rect_pts[1])
    h_src = np.linalg.norm(rect_pts[0] - rect_pts[3])
    
    if h_src > w_src:
        rect_pts = np.array([rect_pts[3], rect_pts[0], rect_pts[1], rect_pts[2]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    warped = cv2.warpPerspective(img_np, M, (dst_w, dst_h), flags=cv2.INTER_LANCZOS4)
    return Image.fromarray(warped)

# --- UI COMPONENT: Manual Crop ---
def manual_crop_ui(label, key_prefix, image_pil):
    pts_key = f"{key_prefix}_points"
    if pts_key not in st.session_state: st.session_state[pts_key] = []

    w_orig, h_orig = image_pil.size
    display_width = 500
    ratio = display_width / w_orig
    display_height = int(h_orig * ratio)
    img_resized = image_pil.resize((display_width, display_height))
    
    st.markdown(f"**{label} (Chỉnh tay)**")
    st.caption("Click 4 góc -> Bấm Cắt")

    img_draw = img_resized.copy()
    draw = ImageDraw.Draw(img_draw)
    points = st.session_state[pts_key]
    
    for i, p in enumerate(points):
        px, py = int(p[0]*ratio), int(p[1]*ratio)
        color = "#00FF00" if i == 3 else "#FF0000"
        draw.ellipse((px-5, py-5, px+5, py+5), fill=color, outline="white")
        draw.text((px+8, py), str(i+1), fill="yellow")

    value = streamlit_image_coordinates(img_draw, key=f"coord_{key_prefix}", width=display_width)

    if value:
        real_x = value["x"] / ratio
        real_y = value["y"] / ratio
        
        # Check duplicate click
        is_dup = False
        if points:
            last = points[-1]
            if abs(last[0]-real_x) < 5 and abs(last[1]-real_y) < 5: is_dup = True
        
        if not is_dup and len(points) < 4:
            points.append((real_x, real_y))
            st.session_state[pts_key] = points
            st.rerun()

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("🗑️ Xóa", key=f"del_{key_prefix}"):
            st.session_state[pts_key] = []
            st.rerun()
    with c2:
        if len(points) == 4:
            if st.button("✂️ Cắt Ngay", key=f"cut_{key_prefix}", type="primary"):
                return warp_from_points(image_pil, points)
    return None

# --- MAIN APP ---

def main():
    st.markdown("<h1 style='text-align: center; color: #d35400;'>🆔 TOOL CCCD V19 (HYBRID)</h1>", unsafe_allow_html=True)
    
    # --- 1. CHỌN CHẾ ĐỘ ---
    st.markdown("### ⚙️ Cài đặt chế độ")
    mode = st.radio("Chọn cách xử lý:", ["🤖 Tự động (AI Auto)", "🖐️ Chỉnh tay (Click 4 điểm)"], horizontal=True)
    
    # Load AI nếu cần
    session = None
    if "Tự động" in mode:
        with st.spinner("Đang tải AI..."):
            session = load_ai_session()

    # --- 2. UPLOAD ẢNH ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1: f_file = st.file_uploader("Mặt Trước", type=['jpg','png','jpeg'], key="f")
    with col2: b_file = st.file_uploader("Mặt Sau", type=['jpg','png','jpeg'], key="b")

    img1_final = None
    img2_final = None

    # --- 3. XỬ LÝ ẢNH ---
    if f_file and b_file:
        img1 = Image.open(f_file)
        img2 = Image.open(b_file)

        # A. CHẾ ĐỘ TỰ ĐỘNG
        if "Tự động" in mode:
            if st.button("🚀 XỬ LÝ TỰ ĐỘNG NGAY", type="primary", use_container_width=True):
                with st.spinner("AI đang xử lý..."):
                    img1_final = auto_process_image(img1, session)
                    img2_final = auto_process_image(img2, session)
                    
                    # Lưu vào session để không bị mất khi rerun
                    st.session_state['res_auto_1'] = img1_final
                    st.session_state['res_auto_2'] = img2_final
            
            # Load lại kết quả nếu đã có
            if 'res_auto_1' in st.session_state:
                img1_final = st.session_state['res_auto_1']
                img2_final = st.session_state['res_auto_2']
                
                # Hiển thị kết quả con
                c1, c2 = st.columns(2)
                with c1: st.image(img1_final, caption="Mặt Trước (Auto)", width=300)
                with c2: st.image(img2_final, caption="Mặt Sau (Auto)", width=300)

        # B. CHẾ ĐỘ CHỈNH TAY
        else:
            c1, c2 = st.columns(2)
            with c1:
                res1 = manual_crop_ui("Mặt Trước", "f_man", img1)
                if res1: st.session_state['res_man_1'] = res1
                if 'res_man_1' in st.session_state: 
                    st.image(st.session_state['res_man_1'], width=300, caption="Đã cắt")
                    img1_final = st.session_state['res_man_1']
            
            with c2:
                res2 = manual_crop_ui("Mặt Sau", "b_man", img2)
                if res2: st.session_state['res_man_2'] = res2
                if 'res_man_2' in st.session_state: 
                    st.image(st.session_state['res_man_2'], width=300, caption="Đã cắt")
                    img2_final = st.session_state['res_man_2']

    # --- 4. GHÉP A4 & XEM TRƯỚC (PREVIEW) ---
    if img1_final and img2_final:
        st.markdown("---")
        st.subheader("📄 KẾT QUẢ CUỐI CÙNG (A4 Preview)")
        
        # Xử lý ghép
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
        
        # --- HIỂN THỊ PREVIEW A4 ---
        st.image(canvas, caption="Bản xem trước A4 (Đã sẵn sàng in)", use_container_width=True, output_format="JPEG")
        
        # --- NÚT TẢI XUỐNG ---
        pdf_buffer = io.BytesIO()
        canvas.save(pdf_buffer, "PDF", resolution=300.0)
        
        st.download_button(
            label="📥 TẢI FILE PDF A4 VỀ MÁY",
            data=pdf_buffer.getvalue(),
            file_name="CCCD_V19_Hybrid.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey;'>App created by Cà Văn Kim - ATP</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()