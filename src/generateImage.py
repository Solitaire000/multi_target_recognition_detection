from pathlib import Path

import cv2
import numpy as np
import os
import random


def generateImage():
    # === 读取图片 ===
    # 当前文件路径
    current_path = Path(__file__).resolve()
    parent_dir = current_path.parent.parent.parent
    A = cv2.imread(parent_dir/"data/crop/A.png")
    B = cv2.imread(parent_dir/"data/crop/B.png", cv2.IMREAD_UNCHANGED)
    C = cv2.imread(parent_dir/"data/crop/C.png", cv2.IMREAD_UNCHANGED)
    D = cv2.imread(parent_dir/"data/crop/D.png", cv2.IMREAD_UNCHANGED)

    hA, wA = A.shape[:2]
    hB, wB = B.shape[:2]
    hC, wC = C.shape[:2]
    hD, wD = D.shape[:2]

    # 缩放

    # === 输出目录 ===
    output_dir = "data/generateImage"
    os.makedirs(output_dir, exist_ok=True)

    # === 生成1000张 ===
    num_images = 1000

    for i in range(num_images):
        canvas = A.copy()

        # 随机位置（允许超出边界）
        xB = random.randint(-wB, wA*0.3)
        yB = random.randint(-hB, hA)

        xC = random.randint(-wC, wA)
        yC = random.randint(-hC, hA)

        xD = random.randint(-wD, wA)
        yD = random.randint(-hD, hA)

        # 先放 B
        canvas = overlay_image(canvas, B, xB, yB)
        canvas = overlay_image(canvas, D, xD, yD)

        # 再放 C（保证在上层）
        canvas = overlay_image(canvas, C, xC, yC)

        # 保存
        save_path = os.path.join(output_dir, f"img_{i:04d}.jpg")
        cv2.imwrite(save_path, canvas)

        if i % 100 == 0:
            print(f"已生成 {i} 张")

    print("生成完成！")


def overlay_image(bg, fg, x, y):
    h_bg, w_bg = bg.shape[:2]
    h_fg, w_fg = fg.shape[:2]

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w_fg, w_bg)
    y2 = min(y + h_fg, h_bg)

    if x1 >= x2 or y1 >= y2:
        return bg

    fg_x1 = x1 - x
    fg_y1 = y1 - y
    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)

    roi_bg = bg[y1:y2, x1:x2]
    roi_fg = fg[fg_y1:fg_y2, fg_x1:fg_x2]

    if fg.shape[2] == 4:
        alpha = roi_fg[:, :, 3] / 255.0
        for c in range(3):
            roi_bg[:, :, c] = (1 - alpha) * roi_bg[:, :, c] + alpha * roi_fg[:, :, c]
    else:
        roi_bg[:] = roi_fg

    bg[y1:y2, x1:x2] = roi_bg
    return bg


def augment_image(input_path, output_dir, num_images=500,index = 0):
    """
    对单张图片进行多种变换，生成增强数据集

    参数：
        input_path: 原始图片路径
        output_dir: 输出目录
        num_images: 生成图片数量（默认500）
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(input_path)

    h, w = img.shape[:2]

    # 图片转换
    def random_transform(image):
        h, w = image.shape[:2]
        transformed = image.copy()

        # 缩放（轻微）
        if random.random() < 0.6:
            scale = random.uniform(0.9, 1.1)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(transformed, (new_w, new_h))

            # 中心裁剪或填充
            if scale > 1:
                start_x = (new_w - w) // 2
                start_y = (new_h - h) // 2
                transformed = resized[start_y:start_y + h, start_x:start_x + w]
            else:
                pad_x = (w - new_w) // 2
                pad_y = (h - new_h) // 2
                transformed = cv2.copyMakeBorder(
                    resized, pad_y, h - new_h - pad_y, pad_x, w - new_w - pad_x,
                    borderType=cv2.BORDER_REFLECT
                )

        # 平移（轻微）
        if random.random() < 0.6:
            tx = random.randint(-10, 10)
            ty = random.randint(-10, 10)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            transformed = cv2.warpAffine(transformed, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 翻转
        if random.random() < 0.4:
            transformed = cv2.flip(transformed, 1)

        # 亮度（轻微）
        if random.random() < 0.9:
            brightness = random.uniform(0.9, 1.1)
            transformed = np.clip(transformed * brightness, 0, 255).astype(np.uint8)

        # 对比度（轻微）
        if random.random() < 0.9:
            contrast = random.uniform(0.9, 1.1)
            transformed = np.clip(contrast * transformed, 0, 255).astype(np.uint8)

        # 高斯噪声（修正版）
        if random.random() < 0.5:
            noise = np.random.normal(0, 5, transformed.shape)
            transformed = transformed.astype(np.float32) + noise
            transformed = np.clip(transformed, 0, 255).astype(np.uint8)

        #  模糊（轻微）
        if random.random() < 0.3:
            k = random.choice([3])
            transformed = cv2.GaussianBlur(transformed, (k, k), 0)

        # 颜色偏移（轻微）
        if random.random() < 0.6:
            shift = np.random.randint(-10, 10, 3)
            transformed = transformed.astype(np.int16) + shift
            transformed = np.clip(transformed, 0, 255).astype(np.uint8)

        return transformed

    # 批量生成
    for i in range(num_images):
        aug_img = random_transform(img)
        output_path = os.path.join(output_dir, f"{index+i}.png")
        cv2.imwrite(output_path, aug_img)
        print(f"{index+i}.png 已经生成")

    print(f"已生成 {num_images} 张图片到 {output_dir}")
