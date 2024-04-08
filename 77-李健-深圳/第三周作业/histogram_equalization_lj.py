# 直方图均衡化
import cv2
import numpy
from matplotlib import pyplot as plt

'''均衡化'''
def histogram_equalization():
    img = cv2.imread("lenna.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 调均衡化方法
    dtc = cv2.equalizeHist(gray)
    # hist = cv2.calcHist([dtc], [0], None, [256], [0, 256])
    plt.figure()
    plt.hist(dtc.ravel(), 256)
    plt.show()
    cv2.imshow("img", numpy.hstack([dtc, gray]))
    cv2.waitKey(0)


def color_equalization():
    src_img = cv2.imread("F:\pythonProject\imageProject\pythonProject\lenna.png")
    # cv2.imshow("原图", src_img)
    #彩图均衡化，需将三通道分别均衡化再合并
    b, g, r = cv2.split(src_img)
    new_b = cv2.equalizeHist(b)
    new_g = cv2.equalizeHist(g)
    new_r = cv2.equalizeHist(r)
    #均衡化后合并
    new_img = cv2.merge((new_b, new_g, new_r))
    cv2.imshow("new_ing", numpy.hstack([src_img,new_img]))
    cv2.waitKey(0)


color_equalization()

histogram_equalization()
