import cv2
import numpy as np


# 特征匹配
def feature_matching(img_1, kp1, img_2, kp2, goodMatch):
    h1, w1 = img_1.shape[:2]
    h2, w2 = img_2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img_1
    vis[:h2, w1:w1 + w2] = img_2
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]
    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)
    # 将每一组相匹配的特征点用线连接
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
    # 用于创建一个窗口
    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)


img_1 = cv2.imread('iphone1.png')
img_2 = cv2.imread('iphone2.png')
sift = cv2.xfeatures2d.SIFT_create()
# 返回关键点和描述子
kp1, des1 = sift.detectAndCompute(img_1, None)
kp2, des2 = sift.detectAndCompute(img_2, None)
# 交叉匹配
bf = cv2.BFMatcher(cv2.NORM_L2)
# 匹配特征点
matches = bf.knnMatch(des1, des2, k=2)
# queryIdx 莫一特征点在本图上的索引
# trainIdx 该特征点在另一张图片中相匹配的特征点的索引
# distance 表示这一对特征点的欧式距离，数值越小说明两个特征点越相近
goodMatch = []
for m, n in matches:
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)

# img = cv2.drawMatches(img_1, kp1, img_2, kp2, goodMatch[:20], img_2, flags=2)
# cv2.imshow('img', img)
feature_matching(img_1, kp1, img_2, kp2, goodMatch)
cv2.waitKey(0)
cv2.destroyAllWindows()
