import cv2
import numpy as np

'''双线性插值'''


def bilinear_interpolation_lj(img, out_img):
    h, w, c = img.shape
    dit_h = out_img[0]
    dit_w = out_img[1]
    if h == dit_h and dit_w == w:
        return img.copy
    dit_img = np.zeros((dit_h, dit_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(w) / dit_w, float(h) / dit_h
    for i in range(c):
        for dit_y in range(dit_h):
            for dit_x in range(dit_w):
                src_x = (dit_x + 0.5) * scale_x - 0.5
                src_y = (dit_y + 0.5) * scale_y - 0.5
                # 取得到的目标图的原坐标对的相邻的四个点的值，因为步长都为一 所以取整即可
                src_x1 = int(np.floor(src_x))
                src_x2 = min(src_x1 + 1, w - 1)
                src_y1 = int(np.floor(src_y))
                src_y2 = min(src_y1 + 1, h - 1)
                # 套用公式
                r1 = (src_x2 - src_x) * img[src_y1, src_x1, i] + (src_x - src_x1) * img[src_y1, src_x2, i]
                r2 = (src_x2 - src_x) * img[src_y2, src_x1, i] + (src_x - src_x1) * img[src_y2, src_x2, i]
                dit_img[dit_y, dit_x, i] = int((src_y2 - src_y) * r1 + (src_y - src_y1) * r2)

    return dit_img


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    dit_img = bilinear_interpolation_lj(img, (700, 700))
    cv2.imshow("src_img", img)
    cv2.imshow("new_img", dit_img)
    cv2.waitKey(0)
