import time

import cv2

image = cv2.imread("images/photo/sd_raw_1701362292.png")
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
scharr = cv2.Canny(gray, 10, 50)
cv2.imshow("Scharr", scharr)
cv2.waitKey(0)