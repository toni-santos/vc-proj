import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = "samples/0_hWqG_original_3460_1609895876431.jpg"

img = cv2.imread(img_path)
im = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

v = np.median(img_gray)
sigma = 0.33
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))

img_canny = cv2.Canny(img, lower, upper)

thresh = cv2.threshold(img_canny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = contours[0] if len(contours) == 2 else contours[1]
for cntr in contours:
    x,y,w,h = cv2.boundingRect(cntr)
    cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 2)
    print("x,y,w,h:",x,y,w,h)

cv2.imwrite('img_canny.jpg', img_canny)
cv2.imwrite('im.jpg', im)