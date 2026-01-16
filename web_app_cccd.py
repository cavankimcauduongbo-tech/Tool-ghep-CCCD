import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from rembg import remove, new_session
import io
import gc

# --- CẤU HÌNH ---
st.set_page_config(page_title="Tool CCCD V15 (Chỉnh Phối Cảnh)", page_icon="🆔", layout="centered")

# --- CORE LOGIC ---

@st.cache_resource
def load_ai_session():
    return new_session("u2netp")

def pixel_from_mm(mm, dpi=300):
    return int(mm * dpi / 25.4)

def get_default_points(img_pil, session):
    """
    Dùng AI đoán trước 4 góc để bạn đỡ phải kéo nhiều.
    Trả về dict chứa 4 cặp tọa độ: TL, TR, BR, BL
    """
    w, h = img_pil.size
    # Mặc định là 4 góc ảnh lùi vào 10%
    pad_x = int(w * 0.1)
    pad_y = int(h * 0.1)
    
    default_pts = {
        "tl": [pad_x, pad_y],          # Top-Left
        "tr": [w - pad_x, pad_y],      # Top-Right
        "br": [w - pad_x, h - pad_y],  # Bot-Right
        "bl": [pad_x, h - pad_y]       # Bot-Left
    }

    try:
        # Resize nhỏ để AI chạy nhanh
        img_np = np.array(img_pil.convert("RGB"))
        small = cv2.resize(img_np, (0,0), fx=0.5, fy=0.5)
        small_pil = Image.fromarray(small)
        
        mask = remove(small_pil, session=session, only_mask=True)
        mask = np.array(mask)
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            
            if len(approx) == 4:
                # Map lại về kích thước gốc (nhân 2)
                pts = approx.reshape(4, 2) * 2
                
                # Sắp xếp điểm
                s = pts.sum(axis=1)
                diff = pts[:, 0] - pts[:, 1] # x - y (Logic khác xíu)
                # Logic sắp xếp thủ công an toàn hơn
                # TL: tổng nhỏ nhất, BR: tổng lớn nhất
                # TR: hiệu x-y lớn nhất, BL: hiệu x-y nhỏ nhất (hoặc ngược lại tùy hệ trục)
                # Dùng logic sort theo Y rồi theo X cho chắc
                
                # Cách sắp xếp đơn giản nhất:
                # Top: 2 điểm có Y nhỏ nhất -> Trong đó X nhỏ là TL, X lớn là TR
                # Bot: 2 điểm có Y lớn nhất -> Trong đó X nhỏ là BL, X lớn là BR
                pts = pts[pts[:, 1].argsort()] # Sort theo Y
                top = pts[:2]
                bot = pts[2:]
                
                top = top[top[:, 0].argsort()] # Sort theo X
                bot = bot[bot[:, 0].argsort()]
                
                default_pts["tl"] = top[0].tolist()
                default_pts["tr"] = top[1].tolist()
                default_pts["bl"] = bot[0].tolist()
                default_pts["br"] = bot[1].tolist()

    except:
        pass # Nếu AI lỗi thì dùng mặc định
        
    return default_pts

def warp_perspective_manual(img_pil, pts_dict):
    """
    Biến đổi phối cảnh từ 4 điểm người dùng chọn
    """
    img_np = np.array(img_pil.convert("RGB"))
    
    # 4 điểm nguồn từ người dùng
    src_pts = np.array([
        pts_dict["tl"],
        pts_dict["tr"],
        pts_dict["br"],
        pts_dict["bl"]
    ], dtype="float32")
    
    # 4 điểm đích (Kích thước chuẩn ID-1 300DPI)
    # 85.6mm x 53.98mm => 1011 x 638 pixel
    dst_w, dst_h = 1011, 638
    
    dst_pts = np.array([
        [0, 0],           # TL
        [dst_w - 1, 0],   # TR
        [dst_w - 1, dst_h - 1], # BR
        [0, dst_h - 1]    # BL
    ], dtype="float32")
    
    # Tính ma trận biến đổi
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Warp (ép phẳng)
    warped = cv2.warpPerspective(img_np, M, (dst_w, dst_h), flags=cv2.INTER_LANCZOS4)
    
    return Image.fromarray(warped)

def draw_guide(img_pil, pts):
    """Vẽ khung nối 4 điểm để người dùng căn chỉnh"""
    draw_img = img_pil.copy()
    draw = ImageDraw.Draw(draw_img)
    
    p = [tuple(pts["tl"]), tuple(pts["tr"]), tuple(pts["br"]), tuple(pts["bl"])]
    
    # Vẽ đa giác nối
    draw.polygon(p, outline="#00FF00", width=4)
    
    # Vẽ chấm tròn to rõ
    r = 15
    # TL - Đỏ
    draw.ellipse((p[0][0]-r, p[0][1]-r, p[0][0]+r, p[0][1]+r), fill="red", outline="white", width=2)
    # TR - Xanh lá
    draw.ellipse((p[1][0]-r, p[1][1]-r, p[1][0]+r, p[1][1]+r), fill="green", outline="white", width=2)
    # BR - Xanh dương
    draw.ellipse((p[2][0]-r, p[2][1]-r, p[2][0]+r, p[2][1]+r), fill="blue", outline="white", width=2)
    # BL - Vàng
    draw.ellipse((p[3][0]-r, p[3][1]-r, p[3][0]+r, p[3][1]+r), fill="yellow", outline="white", width=2)
    
    return draw_img

# --- UI COMPONENT ---

def adjustment_ui(label, key_prefix, uploaded_file, session):
    if not uploaded_file: return None
    
    img = Image.open(uploaded_file)
    w, h = img.size
    
    # Khởi tạo tọa độ 1 lần
    state_key = f"{key_prefix}_pts"
    if state_key not in st.session_state:
        st.session_state[state_key] = get_default_points(img, session)
        
    pts = st.session_state[state_key]
    
    st.markdown(f"#### 📐 Căn chỉnh {label}")
    st.info("Kéo thanh trượt sao cho 4 chấm màu nằm đúng 4 góc nhọn của thẻ.")

    # Hiển thị ảnh hướng dẫn
    preview = draw_guide(img, pts)
    st.image(preview, use_container_width=True)
    
    # Sliders điều khiển
    # Chia làm 2 cột: Trái và Phải
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Bên Trái**")
        st.markdown("🔴 **Góc Trên-Trái (Đỏ)**")
        pts["tl"][0] = st.slider(f"X (Trái-Phải)", 0, w, pts["tl"][0], key=f"{key_prefix}_tl_x")
        pts["tl"][1] = st.slider(f"Y (Lên-Xuống)", 0, h, pts["tl"][1], key=f"{key_prefix}_tl_y")
        
        st.markdown("🟡 **Góc Dưới-Trái (Vàng)**")
        pts["bl"][0] = st.slider(f"X (Trái-Phải)", 0, w, pts["bl"][0], key=f"{key_prefix}_bl_x")
        pts["bl"][1] = st.slider(f"Y (Lên-Xuống)", 0, h, pts["bl"][1], key=f"{key_prefix}_bl_y")

    with c2:
        st.markdown("**Bên Phải**")
        st.markdown("🟢 **Góc Trên-Phải (Xanh Lá)**")
        pts["tr"][0] = st.slider(f"X (Trái-Phải)", 0, w, pts["tr"][0], key=f"{key_prefix}_tr_x")
        pts["tr"][1] = st.slider(f"Y (Lên-Xuống)", 0, h, pts["tr"][1], key=f"{key_prefix}_tr_y")
        
        st.markdown("🔵 **Góc Dưới-Phải (Xanh Dương)**")
        pts["br"][0] = st.slider(f"X (Trái-Phải)", 0, w, pts["br"][0], key=f"{key_prefix}_br_x")
        pts["br"][1] = st.slider(f"Y (Lên-Xuống)", 0, h, pts["br"][1], key=f"{key_prefix}_br_y")

    # Cập nhật lại session
    st.session_state[state_key] = pts
    
    # Xử lý cắt
    final_card = warp_perspective_manual(img, pts)
    return final_card

def main():
    st.markdown("<h1 style='text-align: center; color: #8e44ad;'>🆔 TOOL V15 (CAMSCANNER MODE)</h1>", unsafe_allow_html=True)
    st.caption("Chỉnh 4 góc thủ công - Không bao giờ bị méo")
    
    session = load_ai_session()
    
    col1, col2 = st.columns(2)
    with col1:
        f_file = st.file_uploader("Mặt Trước", type=['jpg','png','jpeg'], key="f_up")
    with col2:
        b_file = st.file_uploader("Mặt Sau", type=['jpg','png','jpeg'], key="b_up")

    # Xử lý từng ảnh
    img1_final = None
    img2_final = None

    if f_file:
        with st.expander("1. CHỈNH SỬA MẶT TRƯỚC", expanded=True):
            img1_final = adjustment_ui("Mặt Trước", "front", f_file, session)
            if img1_final:
                st.image(img1_final, caption="Kết quả Mặt Trước", width=300)

    if b_file:
        with st.expander("2. CHỈNH SỬA MẶT SAU", expanded=False):
            img2_final = adjustment_ui("Mặt Sau", "back", b_file, session)
            if img2_final:
                st.image(img2_final, caption="Kết quả Mặt Sau", width=300)

    # Nút ghép
    if img1_final and img2_final:
        st.markdown("---")
        if st.button("📄 XUẤT FILE PDF", type="primary", use_container_width=True):
            with st.spinner("Đang tạo PDF..."):
                # Kích thước pixel chuẩn
                TARGET_W, TARGET_H = 1011, 638
                
                # Resize (Dù warp đã chuẩn, resize lại lần nữa cho chắc chắn đúng size in)
                scan1 = img1_final.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
                scan2 = img2_final.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)

                # A4 Canvas
                A4_W, A4_H = pixel_from_mm(210, 300), pixel_from_mm(297, 300)
                canvas = Image.new('RGB', (A4_W, A4_H), 'white')
                
                cx = A4_W // 2
                gap = 350
                sy = (A4_H - (TARGET_H * 2 + gap)) // 2 

                canvas.paste(scan1, (cx - TARGET_W // 2, sy))
                canvas.paste(scan2, (cx - TARGET_W // 2, sy + TARGET_H + gap))

                # Save
                pdf_buffer = io.BytesIO()
                canvas.save(pdf_buffer, "PDF", resolution=300.0)
                
                st.success("Thành công!")
                st.image(canvas, caption="File A4 Hoàn Chỉnh", use_container_width=True)
                st.download_button("📥 TẢI PDF NGAY", pdf_buffer.getvalue(), "CCCD_V15_Manual.pdf", "application/pdf", type="primary")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey;'>App created by Cà Văn Kim - ATP</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()