import cv2

#以灰度方式加载图像
img = cv2.imread("lenna.png",0)
sobX = cv2.Sobel(img, cv2.CV_16S, 1, 0)
sobY = cv2.Sobel(img, cv2.CV_16S, 0, 1)

imgX = cv2.convertScaleAbs(sobX)
imgY = cv2.convertScaleAbs(sobY)

resImg = cv2.addWeighted(imgX, 0.5, imgY, 0.5, 0)

cv2.imshow("x", imgX)
cv2.imshow("y", imgY)
cv2.imshow("result", resImg)
cv2.waitKey(0)
cv2.destroyAllWindows()