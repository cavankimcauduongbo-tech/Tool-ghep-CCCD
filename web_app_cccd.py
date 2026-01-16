import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
from rembg import remove, new_session

# --- CẤU HÌNH OFFLINE ---
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

os.environ["U2NET_HOME"] = get_resource_path("ai_models")

try:
    my_session = new_session("u2net")
except:
    my_session = None

# ==========================================
# THUẬT TOÁN V3: AI MASK + PERSPECTIVE WARP
# ==========================================

def order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: TL, TR, BR, BL"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left có tổng (x+y) nhỏ nhất
    # Bottom-right có tổng (x+y) lớn nhất
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right có hiệu (y-x) nhỏ nhất
    # Bottom-left có hiệu (y-x) lớn nhất
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(image, pts):
    """Ép ảnh về hình chữ nhật chuẩn dựa trên 4 điểm"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Tính chiều rộng tối đa
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Tính chiều cao tối đa
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Ma trận đích (Hình chữ nhật vuông vức)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Tính ma trận biến đổi và áp dụng
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def find_corners_from_mask(mask):
    """Tìm 4 góc cực trị từ hình dạng mask"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    
    # Lấy contour lớn nhất
    c = max(cnts, key=cv2.contourArea)
    
    # Tính xấp xỉ đa giác
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    # Nếu xấp xỉ ra đúng 4 điểm thì quá tốt
    if len(approx) == 4:
        return approx.reshape(4, 2)
    
    # Nếu không (do bo góc tròn nên ra nhiều điểm), ta tìm Convex Hull
    # Sau đó tìm 4 điểm sát với 4 góc của bounding rect nhất
    hull = cv2.convexHull(c)
    hull = hull.reshape(-1, 2)
    
    # Logic tìm 4 góc "cực trị" thủ công:
    # TL: x+y min, BR: x+y max, TR: x-y max, BL: x-y min (hoặc logic tương tự)
    # Ở đây dùng cách đơn giản nhất: order_points cho toàn bộ hull rồi lấy 4 điểm đó
    # Tuy nhiên cách tốt nhất là dùng rect xoay để định hướng
    
    # Fallback: Dùng minAreaRect để lấy 4 góc hộp bao, 
    # nhưng như thế vẫn dính viền đen nếu ảnh nghiêng.
    # -> Dùng phương pháp tìm điểm cực:
    
    s = hull.sum(axis=1)
    diff = np.diff(hull, axis=1)
    
    tl = hull[np.argmin(s)]
    br = hull[np.argmax(s)]
    tr = hull[np.argmin(diff)]
    bl = hull[np.argmax(diff)]
    
    return np.array([tl, tr, br, bl], dtype="float32")

def process_scan_v3(image_path, session):
    # 1. Đọc ảnh
    img_cv = cv2.imread(image_path)
    if img_cv is None: return None
    orig = img_cv.copy()
    
    # 2. Dùng AI để lấy Mask (Chỉ lấy hình dáng, không cắt vội)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    try:
        # Lấy mask (alpha channel)
        if session:
            out = remove(img_rgb, session=session, only_mask=True)
        else:
            out = remove(img_rgb, only_mask=True)
            
        # Mask trả về là ảnh xám (Grayscale)
        mask = np.array(out)
        
        # 3. Tìm 4 góc từ Mask
        pts = find_corners_from_mask(mask)
        
        if pts is not None:
            # 4. Ép phẳng (Warp) ảnh gốc theo 4 góc tìm được
            warped = perspective_transform(orig, pts)
            warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            return Image.fromarray(warped_rgb)
        else:
            return Image.fromarray(img_rgb)
            
    except Exception as e:
        print(f"Lỗi: {e}")
        return Image.fromarray(img_rgb)

# ==========================================
# GIAO DIỆN (GIỮ NGUYÊN)
# ==========================================
class CCCDAppV3:
    def __init__(self, root):
        self.root = root
        self.root.title("Tool CCCD - Chế độ Scan Phẳng")
        self.root.geometry("500x750")
        self.root.resizable(False, False)
        
        BG_COLOR = "#f4f4f4"
        self.root.configure(bg=BG_COLOR)
        self.front_path = None
        self.back_path = None
        
        tk.Label(root, text="SCAN CCCD PHẲNG", font=("Arial", 20, "bold"), fg="#c0392b", bg=BG_COLOR).pack(pady=(25, 5))
        tk.Label(root, text="(Ép thành hình chữ nhật chuẩn)", font=("Arial", 10), fg="#555", bg=BG_COLOR).pack(pady=(0, 10))

        self.create_input_frame(1, "Mặt Trước")
        self.create_input_frame(2, "Mặt Sau")

        self.var_ai = tk.BooleanVar(value=True)
        tk.Checkbutton(root, text="Kích hoạt AI Scan (Khuyên dùng)", variable=self.var_ai, bg=BG_COLOR).pack(pady=5)

        self.btn_run = tk.Button(root, text="XỬ LÝ VÀ XUẤT PDF", command=self.process, bg="#c0392b", fg="white", font=("Arial", 12, "bold"), height=2, width=30, relief="flat")
        self.btn_run.pack(pady=20)

        self.lbl_status = tk.Label(root, text="Sẵn sàng", fg="gray", bg=BG_COLOR)
        self.lbl_status.pack(pady=5)

        footer = tk.Frame(root, bg="#2c3e50", height=40)
        footer.pack(side="bottom", fill="x")
        footer.pack_propagate(False)
        tk.Label(footer, text="App created by Cà Văn Kim - ATP", font=("Segoe UI", 10, "bold"), fg="white", bg="#2c3e50").place(relx=0.5, rely=0.5, anchor="center")

    def create_input_frame(self, idx, title):
        f = tk.Frame(self.root, bg="#f4f4f4", highlightbackground="#ccc", highlightthickness=1)
        f.pack(pady=10, padx=25, fill="x")
        tk.Label(f, text=f"{idx}. Ảnh {title}:", font=("Arial", 10, "bold"), bg="#f4f4f4").pack(anchor="w", padx=5, pady=5)
        lbl = tk.Label(f, text="[Chưa chọn ảnh]", bg="#e0e0e0", height=5)
        lbl.pack(fill="x", padx=5, pady=5)
        btn = tk.Button(f, text="Chọn ảnh...", command=lambda: self.select_img(idx, lbl))
        btn.pack(pady=5)
        if idx == 1: self.lbl_front = lbl
        else: self.lbl_back = lbl

    def select_img(self, idx, lbl):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.jpeg")])
        if p:
            if idx == 1: self.front_path = p
            else: self.back_path = p
            img = Image.open(p)
            img.thumbnail((300, 150))
            photo = ImageTk.PhotoImage(img)
            lbl.config(image=photo, text="", height=0)
            lbl.image = photo

    def process(self):
        if not self.front_path or not self.back_path:
            messagebox.showwarning("Thiếu ảnh", "Vui lòng chọn đủ 2 ảnh!")
            return
        out_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if not out_path: return

        self.lbl_status.config(text="Đang quét và ép phẳng ảnh...", fg="blue")
        self.btn_run.config(state="disabled")
        threading.Thread(target=self.run_logic, args=(out_path,)).start()

    def run_logic(self, output_path):
        try:
            DPI = 300
            # Kích thước chuẩn ID-1
            CARD_W, CARD_H = 1011, 638 # Tương đương 85.6mm x 53.98mm tại 300 DPI
            A4_W, A4_H = 2480, 3508 # A4 tại 300 DPI

            if self.var_ai.get():
                img1 = process_scan_v3(self.front_path, my_session)
                img2 = process_scan_v3(self.back_path, my_session)
            else:
                img1 = Image.open(self.front_path).convert("RGB")
                img2 = Image.open(self.back_path).convert("RGB")

            # Force resize về đúng chuẩn hình chữ nhật
            img1 = img1.resize((CARD_W, CARD_H), Image.Resampling.LANCZOS)
            img2 = img2.resize((CARD_W, CARD_H), Image.Resampling.LANCZOS)

            # Dán lên A4
            canvas = Image.new('RGB', (A4_W, A4_H), 'white')
            cx = A4_W // 2
            gap = 300
            sy = (A4_H - (CARD_H * 2 + gap)) // 2 

            canvas.paste(img1, (cx - CARD_W // 2, sy))
            canvas.paste(img2, (cx - CARD_W // 2, sy + CARD_H + gap))

            canvas.save(output_path, "PDF", resolution=300.0)
            
            self.root.after(0, lambda: [
                self.lbl_status.config(text="Hoàn thành!", fg="green"),
                self.btn_run.config(state="normal"),
                messagebox.showinfo("Thành công", f"Đã xuất file: {output_path}")
            ])
        except Exception as e:
            self.root.after(0, lambda: [
                self.lbl_status.config(text="Lỗi!", fg="red"),
                self.btn_run.config(state="normal"),
                messagebox.showerror("Lỗi", str(e))
            ])

if __name__ == "__main__":
    tk.Tk()
    app = CCCDAppV3(tk._default_root)
    tk._default_root.mainloop()