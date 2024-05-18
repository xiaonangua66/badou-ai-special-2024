import cv2
import numpy as np
import matplotlib.pyplot as plt


def k_means_demo():
    # 读取原始图像 灰度
    img = cv2.imread("lenna.png", 0)
    # 获取图像高度，宽度
    row, clos = img.shape[:]
    # 转为一维
    data = img.reshape((row * clos, -1))
    data = np.float32(data)
    # 设置停止条件(type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 设置标签
    flags = cv2.KMEANS_RANDOM_CENTERS
    # K-Means聚类 聚成四类
    compactness, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, flags)
    # 生成最终图像
    dst = labels.reshape((img.shape[0], img.shape[1]))
    # 显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 显示图像
    titles = [u'原始图像', u'聚类图像']
    image = [img, dst]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(image[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.ylabel([])
    plt.show()

k_means_demo()