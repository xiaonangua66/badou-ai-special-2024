import random

import cv2


def gaussian_noise(src_img, mean, sigma, percentage):
    noise_img = src_img
    noise_num = int(src_img.shape[0] * src_img.shape[1] * percentage)
    for i in range(noise_num):
        x = random.randint(0, src_img.shape[0] - 1)
        y = random.randint(0, src_img.shape[1] - 1)
        noise_img[x, y] = noise_img[x, y] + random.gauss(mean, sigma)
        if noise_img[x, y] < 0:
            noise_img[x, y] = 0
        elif noise_img[x, y] > 255:
            noise_img[x, y] = 255
    return noise_img


src_img = cv2.imread("lenna.png", 0)
gaussian_noise_img = gaussian_noise(src_img, 4, 4, 0.8)
img2 = cv2.imread('lenna.png')
gray_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imshow("src_img", gray_img)
cv2.imshow("gaussian_img", gaussian_noise_img)
cv2.waitKey(0)
