import cv2


def sitf_keys():
    img = cv2.imread('lenna.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    #返回关键点和特征描述子
    keypoints, descriptor = sift.detectAndCompute(gray, None)
    # print(descriptor)
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 对图像的每个关键点都绘制圆圈和方向。
    # img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    #                   color=(22, 234, 212))
    img = cv2.drawKeypoints(img, keypoints, None)
    # 展示图片
    cv2.imshow('sift_keypoints', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


sitf_keys()
