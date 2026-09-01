import os
import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
from PIL import Image, ImageTk
import math

# 加载文件夹中的image
def load_images_from_folder(folder_path, exts=(".jpg", ".png", ".bmp")):
    image_paths = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(exts):
            image_paths.append(os.path.join(folder_path, fname))
    return image_paths



def showConnected(mat: np.ndarray,con = 8):
    """
    可视化连通域分析结果：每个簇显示不同颜色
    """
    num_labels, label_ids, stats, centroids = cv2.connectedComponentsWithStats(mat, connectivity=con)
    '''
       num_labels   # 连通域总数（包含背景0）
       labels       # 和原图一样大的矩阵，每个像素值=它属于哪个连通域
       stats        # 形状 [num_labels, 5]
                    # stats[label, 0] = x
                    # stats[label, 1] = y
                    # stats[label, 2] = width
                    # stats[label, 3] = height
                    # stats[label, 4] = area(面积)
       centroids    # 形状 [num_labels, 2] → 每个连通域的中心 (cx, cy)
    '''
    # 创建彩色图
    vis = np.zeros((mat.shape[0], mat.shape[1], 3), dtype=np.uint8)

    for label in range(1, num_labels):  # 0是背景，跳过
        # 随机颜色
        color = np.random.randint(0, 255, size=3).tolist()
        vis[label_ids == label] = color
        # 可选：绘制中心点
        cx, cy = int(centroids[label][0]), int(centroids[label][1])
        cv2.circle(vis, (cx, cy), 2, (255,255,255), -1)
        cv2.putText(vis, str(label), (cx+5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

    showImage("Connected Components (每个簇不同颜色)", vis)
    return vis


def showImage(title: str, mat: np.ndarray) -> None:
    if title is None:
        title = "IMAGE"

    if mat is None or mat.size == 0:
        raise ValueError("showImage: mat 为空或无效")

    if mat.ndim == 2:
        rgb = np.stack([mat, mat, mat], axis=2)
        is_gray = True
    elif mat.ndim == 3 and mat.shape[2] == 3:
        rgb = mat[:, :, ::-1].copy()
        is_gray = False
    elif mat.ndim == 3 and mat.shape[2] == 4:
        rgb = cv2.cvtColor(mat, cv2.COLOR_BGRA2RGBA)
        is_gray = False
    else:
        raise ValueError(f"showImage: 不支持的格式 shape={mat.shape}")

    img_h, img_w = rgb.shape[:2]

    # 金字塔
    MAX_LEVELS = 6
    pyramid = [rgb]
    for _ in range(MAX_LEVELS - 1):
        prev = pyramid[-1]
        ph, pw = prev.shape[:2]
        if pw < 4 or ph < 4:
            break
        down = cv2.resize(prev, (max(1, pw // 2), max(1, ph // 2)),
                          interpolation=cv2.INTER_AREA)
        pyramid.append(down)

    def best_pyramid_level(scale: float):
        for lvl in range(len(pyramid)):
            layer_scale = 1.0 / (2 ** lvl)
            if layer_scale <= scale:
                if lvl > 0:
                    lvl -= 1
                local = scale / (1.0 / (2 ** lvl))
                return lvl, local
        return len(pyramid) - 1, scale / (1.0 / (2 ** (len(pyramid) - 1)))

    # Tk
    if tk._default_root is None:
        root = tk.Tk()
    else:
        root = tk.Toplevel()
    root.title(title)
    root.configure(bg="#1a1a2e")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    max_w, max_h = int(sw * 0.9), int(sh * 0.85)
    init_scale = min(max_w / img_w, max_h / img_h, 1.0)
    canvas_w = min(max(int(img_w * init_scale), 400), max_w)
    canvas_h = min(max(int(img_h * init_scale), 300), max_h)

    info_frame = tk.Frame(root, bg="#0f0f23", height=28)
    info_frame.pack(fill=tk.X, side=tk.TOP)
    info_frame.pack_propagate(False)
    tk.Label(info_frame,
             text=f"  {img_w} x {img_h}  |  {'灰度' if is_gray else 'BGR 彩色'}",
             bg="#0f0f23", fg="#6c7a8a", font=("Consolas", 9), anchor="w"
             ).pack(side=tk.LEFT, padx=6)
    tk.Label(info_frame,
             text="拖拽平移  |  滚轮缩放  |  R 重置  |  S 保存  |  Q/Esc 关闭  | Space 继续",
             bg="#0f0f23", fg="#444c5c", font=("Consolas", 9), anchor="e"
             ).pack(side=tk.RIGHT)

    canvas = tk.Canvas(root, width=canvas_w, height=canvas_h,
                       bg="#1a1a2e", cursor="crosshair", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    status_frame = tk.Frame(root, bg="#0f0f23", height=24)
    status_frame.pack(fill=tk.X, side=tk.BOTTOM)
    status_frame.pack_propagate(False)
    coord_label = tk.Label(status_frame, text="  就绪",
                           bg="#0f0f23", fg="#58a6ff",
                           font=("Consolas", 9), anchor="w")
    coord_label.pack(side=tk.LEFT, padx=6)
    zoom_label = tk.Label(status_frame, text=f"缩放: {init_scale * 100:.0f}%  ",
                          bg="#0f0f23", fg="#3fb950",
                          font=("Consolas", 9), anchor="e")
    zoom_label.pack(side=tk.RIGHT)

    state = {
        "scale": init_scale,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "drag_sx": 0, "drag_sy": 0,
        "drag_ox": 0.0, "drag_oy": 0.0,
        "tk_img": None,
        "canvas_img_id": None,
        "debounce_id": None,
        "title": None
    }

    MIN_SCALE, MAX_SCALE = 0.01, 64.0
    DEBOUNCE_MS = 60

    # ─────────────────────────────────────────────────────────────
    # ✅ 修复：强制高质量渲染，启动不模糊
    # ─────────────────────────────────────────────────────────────
    def render(quality: bool = True):
        scale = state["scale"]
        ox = state["offset_x"]
        oy = state["offset_y"]
        cw = canvas.winfo_width() or canvas_w
        ch = canvas.winfo_height() or canvas_h

        vp_x0 = max(0.0, ox)
        vp_y0 = max(0.0, oy)
        vp_x1 = min(float(cw), ox + img_w * scale)
        vp_y1 = min(float(ch), oy + img_h * scale)

        if vp_x1 <= vp_x0 or vp_y1 <= vp_y0:
            return

        src_x0 = int(max(0, (vp_x0 - ox) / scale))
        src_y0 = int(max(0, (vp_y0 - oy) / scale))
        src_x1 = int(min(img_w, math.ceil((vp_x1 - ox) / scale)))
        src_y1 = int(min(img_h, math.ceil((vp_y1 - oy) / scale)))

        if src_x1 <= src_x0 or src_y1 <= src_y0:
            return

        # ✅ 修复：强制使用清晰金字塔层
        lvl, _ = best_pyramid_level(scale)
        layer = pyramid[lvl]
        shrink = 2 ** lvl

        lx0 = max(0, src_x0 // shrink)
        ly0 = max(0, src_y0 // shrink)
        lx1 = min(layer.shape[1], (src_x1 + shrink - 1) // shrink)
        ly1 = min(layer.shape[0], (src_y1 + shrink - 1) // shrink)

        if lx1 <= lx0 or ly1 <= ly0:
            return

        crop = layer[ly0:ly1, lx0:lx1]

        dst_w = int(round((src_x1 - src_x0) * scale))
        dst_h = int(round((src_y1 - src_y0) * scale))
        dst_w = max(dst_w, 1)
        dst_h = max(dst_h, 1)

        # ✅ 修复：清晰插值
        # if scale < 1.0:
        #     resized = cv2.resize(crop, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
        # else:
        #     resized = cv2.resize(crop, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
        if scale >= 0.95:
            interp = cv2.INTER_NEAREST  # 像素块清晰
        else:
            interp = cv2.INTER_AREA  # 缩小时清晰

        resized = cv2.resize(crop, (dst_w, dst_h), interpolation=interp)
        tk_img = ImageTk.PhotoImage(Image.fromarray(resized))
        state["tk_img"] = tk_img

        paste_x = int(vp_x0)
        paste_y = int(vp_y0)

        if state["canvas_img_id"] is None:
            state["canvas_img_id"] = canvas.create_image(
                paste_x, paste_y, anchor=tk.NW, image=tk_img)
        else:
            canvas.coords(state["canvas_img_id"], paste_x, paste_y)
            canvas.itemconfig(state["canvas_img_id"], image=tk_img)

        zoom_label.config(text=f"缩放: {scale * 100:.0f}%  ")

    def schedule_hq():
        if state["debounce_id"] is not None:
            root.after_cancel(state["debounce_id"])
        state["debounce_id"] = root.after(DEBOUNCE_MS, lambda: render(True))

    # 修复：先居中 → 再高清渲染
    def on_space(e):
        state["continue_flag"] = True
        root.quit()

    def center_image():
        cw = canvas.winfo_width() or canvas_w
        ch = canvas.winfo_height() or canvas_h
        state["offset_x"] = (cw - img_w * state["scale"]) / 2
        state["offset_y"] = (ch - img_h * state["scale"]) / 2
        render(quality=True)  # 强制高清

    def on_press(event):
        state["drag_sx"] = event.x
        state["drag_sy"] = event.y
        state["drag_ox"] = state["offset_x"]
        state["drag_oy"] = state["offset_y"]
        canvas.config(cursor="fleur")

    def on_drag(event):
        state["offset_x"] = state["drag_ox"] + event.x - state["drag_sx"]
        state["offset_y"] = state["drag_oy"] + event.y - state["drag_sy"]
        render(False)
        schedule_hq()

    def on_release(event):
        canvas.config(cursor="crosshair")
        render(True)

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)

    def zoom(event, delta):
        factor = 1.12 if delta > 0 else 1 / 1.12
        new_scale = max(MIN_SCALE, min(MAX_SCALE, state["scale"] * factor))
        mx, my = event.x, event.y
        state["offset_x"] = mx - (mx - state["offset_x"]) * (new_scale / state["scale"])
        state["offset_y"] = my - (my - state["offset_y"]) * (new_scale / state["scale"])
        state["scale"] = new_scale
        render(False)
        schedule_hq()

    canvas.bind("<MouseWheel>", lambda e: zoom(e, e.delta))
    canvas.bind("<Button-4>", lambda e: zoom(e, 1))
    canvas.bind("<Button-5>", lambda e: zoom(e, -1))


    def on_move(event):
        px_i = int((event.x - state["offset_x"]) / state["scale"])
        py_i = int((event.y - state["offset_y"]) / state["scale"])
        if 0 <= px_i < img_w and 0 <= py_i < img_h:
            if is_gray:
                v = mat[py_i, px_i]
                coord_label.config(
                    text=f"  X={px_i}  Y={py_i}    Gray={v}", fg="#58a6ff")
            else:
                b, g, r = mat[py_i, px_i, :3]
                coord_label.config(
                    text=f"  X={px_i}  Y={py_i}    RGB=({r},{g},{b})  #{r:02X}{g:02X}{b:02X}",
                    fg="#58a6ff")
        else:
            coord_label.config(text="  -", fg="#444c5c")

    canvas.bind("<Motion>", on_move)

    def reset_view(e=None):
        state["scale"] = init_scale
        center_image()

    def save_image(e=None):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")]
        )
        if path:
            cv2.imwrite(path, mat)

    def close(e=None):
        root.quit()  # 先结束 mainloop
        root.destroy()

    root.bind("<r>", reset_view)
    root.bind("<R>", reset_view)
    root.bind("<s>", save_image)
    root.bind("<S>", save_image)
    root.bind("<q>", close)
    root.bind("<Escape>", close)
    root.bind("<space>", on_space)

    _first = [True]

    def on_resize(e):
        if _first[0]:
            _first[0] = False
            center_image()

    canvas.bind("<Configure>", on_resize)

    # ✅ 启动直接高清渲染
    root.after(30, center_image)
    root.mainloop()