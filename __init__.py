import cv2, os
import numpy as np
import matplotlib.pyplot as plt

img_folder = os.listdir('./samples/')
for path in img_folder:
    img_path = f"samples/{path}"

    img = cv2.imread(img_path)

    total_pieces = 0

    MIN_AREA = img.shape[0] * img.shape[1] * 0.001

    im = img.copy()
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    v = np.median(img_gray)

    # attempt to automatically determine the threshold for canny edge
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # TODO: maybe don't hardcode the values?
    img_canny = cv2.Canny(img, 50, 100)

    kernel = np.ones((5,5),np.uint8)
    img_canny = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)

    thresh = cv2.threshold(img_canny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[0] if len(contours) == 2 else contours[1]
    for idx, cntr in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cntr)
        area = w * h

        if area < MIN_AREA:
            continue

        for n_idx, n_cntr in enumerate(contours):
            if idx != n_idx:
                n_area = cv2.contourArea(n_cntr)
                x1,y1,w1,h1 = cv2.boundingRect(n_cntr)
                cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 2)
                total_pieces += 1

    cv2.imwrite(f'output/canny_{path}', img_canny)
    cv2.imwrite(f'output/{path}', im)