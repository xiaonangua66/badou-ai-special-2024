import cv2
import numpy as np
import matplotlib.pyplot as plt


def k_meams_rgb_demo():
    # 读取图片
    img = cv2.imread('lenna.png')
    img2 = k_means_num(2, img)
    img4 = k_means_num(4, img)
    img8 = k_means_num(8, img)
    img16 = k_means_num(16, img)
    img32 = k_means_num(32, img)

    # 原图像转为RGB显示
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图片
    titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4',
              u'聚类图像 K=8', u'聚类图像 K=16', u'聚类图像 K=32']
    images = [img, img2, img4, img8, img16, img32]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


# 聚类公共方法
def k_means_num(k_num, img):
    # 二维转一维
    data = img.reshape((-1, 3))
    data = np.float32(data)

    # 设置停止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # 设置初始中心点
    flags = cv2.KMEANS_RANDOM_CENTERS
    # 开始聚类
    compactness, labels, center = cv2.kmeans(data, k_num, None, criteria, 10, flags)

    # 图像转换为uint8二维类型
    center2 = np.uint8(center)
    res = center2[labels.flatten()]
    dst = res.reshape(img.shape)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    return dst


k_meams_rgb_demo()
