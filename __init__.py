import cv2, os
import numpy as np
import matplotlib.pyplot as plt

img_folder = os.listdir('./easy_samples/')
for path in img_folder:
    img_path = f"samples/{path}"
    total_pieces = 0

    img = cv2.imread(img_path)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    im = img.copy()

    MIN_AREA = img.shape[0] * img.shape[1] * 0.001

    kernel = np.ones((5, 5), np.uint8) 

    # TODO: find better contrast algorithm
    # for y in range(img.shape[0]):
    #     for x in range(img.shape[1]):
    #         for c in range(img.shape[2]):
    #             img[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)

    img = cv2.GaussianBlur(img, (5, 5), 0)

    # attempt to automatically determine the threshold for canny edge
    v = np.median(img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # TODO: maybe don't hardcode the values?
    img_canny = cv2.Canny(img, 50, 100)

    img_canny = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)

    thresh = cv2.threshold(img_canny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for idx, cntr in enumerate(contours):

        x,y,w,h = cv2.boundingRect(cntr)
        area = w * h

        if area < MIN_AREA:
            continue

        draw = True

        for n_idx, n_cntr in enumerate(contours):
            x1,y1,w1,h1 = cv2.boundingRect(n_cntr)
            n_area = cv2.contourArea(n_cntr)
            
            if idx != n_idx and (x >= x1 and y >= y1 and x+w <= x1+w1 and y+h <= y1+h1):
                draw = False
                break

        if draw:
            total_pieces += 1
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 2)

    print(f"Total pieces: {total_pieces}")

    cv2.imwrite(f'easy_output/canny_{path}', img_canny)
    cv2.imwrite(f'easy_output/{path}', im)
