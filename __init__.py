import cv2, os
import numpy as np
import matplotlib.pyplot as plt

img_folder = os.listdir('./samples/')
for path in img_folder:
    img_path = f"samples/{path}"
    total_pieces = 0
    kernel = np.ones((5, 5), np.uint8) 
    kernel2 = np.ones((13, 13), np.uint8) 

    img = cv2.imread(img_path)
    # img = cv2.resize(img, (0, 0), fx=0.70, fy=0.70)
    im = img.copy()
    MIN_AREA = img.shape[0] * img.shape[1] * 0.001

    img = cv2.fastNlMeansDenoisingColored(img,None,10,7,21)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    gamma = 2
    invGamma = 1/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                    for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table, img)

    img = cv2.bilateralFilter(img,9,img.shape[0]//25,img.shape[0]//25)

    img_canny = cv2.Canny(img, 50, 100)

    img_canny = cv2.morphologyEx(img_canny, cv2.MORPH_DILATE, kernel, iterations=1)
    img_canny = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel, iterations=2)

    thresh = cv2.threshold(img_canny, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

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
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    print(f"Total pieces: {total_pieces}")

    cv2.imwrite(f'output/canny_{path}', img_canny)
    cv2.imwrite(f'output/{path}', img)
