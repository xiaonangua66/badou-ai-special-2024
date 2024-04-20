import random

import cv2


def salt_pepper_noise(img, percentage):
    salt_pepper_img = img
    noise_num = int(salt_pepper_img.shape[0] * salt_pepper_img.shape[1] * percentage)
    for i in range(noise_num):
        x = random.randint(0, salt_pepper_img.shape[0] - 1)
        y = random.randint(0, salt_pepper_img.shape[1] - 1)
        salt_pepper_img[x, y] = random.random()
        if salt_pepper_img[x, y] < 0.5:
            salt_pepper_img[x, y] = 0
        elif salt_pepper_img[x, y] >= 0.5:
            salt_pepper_img[x, y] = 255
    return salt_pepper_img


src_img = cv2.imread("lenna.png", 0)
salt_pepper_noise_img = salt_pepper_noise(src_img, 0.3)
img2 = cv2.imread('lenna.png')
gray_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imshow("src", gray_img)
cv2.imshow('noise_img', salt_pepper_noise_img)
cv2.waitKey(0)
