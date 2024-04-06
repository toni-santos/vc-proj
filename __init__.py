import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import time

# color dictionary
COLORDICT= {
    # "Red": [(0, 60)],
    "Yellow": [(20, 40)],
    "Green": [(41, 70)],
    "Cyan": [(71, 100)],
    "Blue": [(101, 130)],
    "Magenta": [(131, 160)]
}

def getColorName(color):

    color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(color)   # H = 0-179, S = 0-255, V = 0-255

    if v < 30:
        return "Black"
    
    if s < 30:
        if v > 230:
            return "White"
        return "Gray"
    
    for color_name, ranges in COLORDICT.items():
        for start, end in ranges:
            if start <= h <= end:
                return color_name

    return "Red"

# INFO: connectedComponents was previously implemented and part of the solution however it was not used in the final version of the code, keeping it as an artifact 
def connectedComponents(img, MIN_AREA, unprocessed = False):
    output = img.copy()
    largest_area = 0

    objs = []

    if unprocessed:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    numLabels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    
    for i in range(0, numLabels):
        if i == 0:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if (area < MIN_AREA):
            continue

        if (area > largest_area):
            largest_area = area

        objs.append({"x": x, "y": y, "w": w, "h": h, "area": area})
        # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # componentMask = (labels == i).astype("uint8") * 255

    return objs, largest_area

def inCorner(img, rect):
    x, y, w, h = rect
    return (x == 0 and y == 0) or (x + w == img.shape[1] and y == 0) or (y + h == img.shape[0] and x == 0) or (x + w == img.shape[1] and y + h == img.shape[0])

def inContour(target, comp):
    x, y, w, h = target
    x1, y1, w1, h1 = comp

    return x >= x1 and y >= y1 and x+w <= x1+w1 and y+h <= y1+h1

def overlapOverThreshold(rect1, rect2, area_threshold):
    # Two rects overlap by more than 60% of the area of either rectangle?
    x1_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
    y1_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))
    overlap_area = x1_overlap * y1_overlap

    area_rect1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area_rect2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

    total_area = area_rect1 + area_rect2 - overlap_area

    overlap_percentage = max((overlap_area / area_rect1) * 100, (overlap_area / area_rect2) * 100)
    # print(overlap_percentage)

    return overlap_percentage > area_threshold

def overlapUnion(rect1, rect2, threshold):
    if (rect1[0] - rect2[2] >= threshold or 
        rect2[0] - rect1[2] >= threshold or
        rect1[1] - rect2[3] >= threshold or 
        rect2[1] - rect1[3] >= threshold):
        return None

    # print("Merging: ", rect1, rect2)

    x1 = min(rect1[0], rect2[0])
    y1 = min(rect1[1], rect2[1])
    x2 = max(rect1[2], rect2[2])
    y2 = max(rect1[3], rect2[3])
    

    # x y w h
    return [x1, y1, x2-x1, y2-y1]

def getColorBGR(img):
    
    height, width, channels = img.shape[:3]

    pixels = np.reshape(img, (height * width, channels))
    
    pixels = np.float32(pixels)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 3  # number of clusters 
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    main_color_index = unique_labels[np.argmax(counts)]
    main_color = centers[main_color_index]
    
    return main_color

def inColorThreshold(colors, color, threshold):

    for c in colors:
        if max(abs(int(c[0]) - int(color[0])),abs(int(c[1]) - int(color[1])),abs(int(c[2]) - int(color[2]))) < threshold:
            return True
    return False

def preProcessing(img, simple = False):
    print(f"Preprocessing...")
    start_time = time.time()

    # Denoising
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if not simple:
        img = cv2.fastNlMeansDenoising(img,None,23,7,21)
    
    # Histogram equalization
    img = cv2.equalizeHist(img)
    
    # Gamma correction
    gamma = 2
    invGamma = 1/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                    for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table, img)

    # Bilateral filtering
    diagonal = int((img.shape[0] ** 2 + img.shape[1] ** 2) ** 0.5)
    img = cv2.bilateralFilter(img,9, diagonal, diagonal)
    print(f"Preprocessing done in {time.time() - start_time} seconds")
    
    return img

def canny(img, canny_lower, canny_upper, threshold, maxval):
    print("Running Canny edge detection...")
    start_time = time.time()
    kernel = np.ones((5, 5), np.uint8) 

    # Canny edge detection
    img_canny = cv2.Canny(img, canny_lower, canny_upper)

    # Morphological operations
    img_canny = cv2.morphologyEx(img_canny, cv2.MORPH_DILATE, kernel, iterations=1)
    img_canny = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Thresholding
    thresh = cv2.threshold(img_canny, threshold, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = [[x,y,w,h] for x, y, w, h in [cv2.boundingRect(c) for c in contours]]

    print(f"Canny edge detection done in {time.time() - start_time} seconds")
    return contours, img_canny

def findPieces(contours, img, img_og, MIN_AREA, path):
    print("Finding pieces...")
    start_time = time.time()
    copy_og = img_og.copy()

    objs = []
    colors = []
    overlap = True
    draw = True
    MAX_DIST = 20 # TODO: tweak this value
    rects = contours

    for rect in rects:
        if inCorner(img, rect):
            rects.remove(rect)

    while overlap:
        overlap = False
        for idx, rect in enumerate(list(rects)):
            draw = True
            x, y, w, h = rect
            area = w * h

            if idx == 0:
                rects_copy = list(rects)
                rects = []

            if area < MIN_AREA:
                continue

            for n_idx, n_rect in enumerate(rects_copy):
                x1,y1,w1,h1 = n_rect
                n_area = w1 * h1

                if n_area < MIN_AREA or n_idx == idx:
                    continue
                
                # Check if the current contour is inside another contour
                if inContour(rect, n_rect):
                    draw = False
                    break

                # Check if the current contour is overlapping another contour
                if overlapUnion([x, y, x+w, y+h], [x1, y1, x1+w1, y1+h1], MAX_DIST):
                    overlap = True
                    x, y, w, h = overlapUnion([x, y, x+w, y+h], [x1, y1, x1+w1, y1+h1], MAX_DIST)
            
            if draw:
                rects.append([x, y, w, h])

        rects = [list(i) for i in set(tuple(i) for i in rects)]

    test = img_og.copy()
    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(test, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imwrite(f'json_output/rects_{path}', test)

    for rect in rects:
        x, y, w, h = rect

        zone = img_og[y:y+h, x:x+w]
        ZONE_MIN_AREA = zone.shape[1] * zone.shape[0] * 0.01

        # GrabCut
        mask = np.zeros(zone.shape[:2],np.uint8)
        rect = (1,1,zone.shape[1],zone.shape[0])
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        mask, bgdModel, fgdModel = cv2.grabCut(zone,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        zone = zone*mask[:,:,np.newaxis]

        # simple preprocess -> canny -> thresh -> close -> erode -> find contours
        zone = preProcessing(zone, simple=True)

        zone = cv2.Canny(zone, 0, 0)

        thresh = cv2.threshold(zone, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        kernel = np.ones((5, 5), np.uint8)
        erode_kernel = np.ones((9, 9), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, erode_kernel, iterations=3)

        contours_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_ = contours_[0] if len(contours_) == 2 else contours_[1]
        contours_ = [[x,y,w,h] for x, y, w, h in [cv2.boundingRect(c) for c in contours_]]

        for idx, c in enumerate(contours_):
            x_, y_, w_, h_ = c
            area_ = w_ * h_
            skip = False

            if area < ZONE_MIN_AREA:
                continue

            x_ = x + x_ 
            y_ = y + y_

            cv2.rectangle(copy_og, (x_, y_), (x_ + w_, y_ + h_), (0, 0, 255), 1)

            objs.append({
                "xmin": x_,
                "ymin": y_,
                "xmax": x_ + w_,
                "ymax": y_ + h_
            })

            cropped = img_og[y_:y_+h_, x_:x_+w_]
            piece_color = list(getColorBGR(cropped))
            split = img_og[0:1,0:1]
            split = np.array([[piece_color]])
            # cv2.namedWindow('color', cv2.WINDOW_NORMAL)
            # cv2.imshow('color', split)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            print(split)
            color = getColorName(split)

            if color not in colors:
                colors.append(color)    

    print(f"Found {len(objs)} pieces and {len(colors)} ({colors}) colors in {time.time() - start_time} seconds")

    return colors, objs, copy_og

def run(path):
    print(f"Running on {path}")

    img_path = f"samples/{path}"

    # TODO: tweak these values
    canny_lower = 50
    canny_upper = 70
    threshold = 0
    maxval = 255

    # Read image
    img = cv2.imread(img_path)
    img_og = img.copy()

    IMG_AREA = img.shape[0] * img.shape[1]
    MAX_AREA = IMG_AREA * 0.5
    MIN_AREA = IMG_AREA * 0.004

    img = preProcessing(img)

    contours, img_canny = canny(img, canny_lower, canny_upper, threshold, maxval)
    # cv2.imwrite(f'json_output/canny_{path}', img_canny)

    colors, objs, img_pieces = findPieces(contours, img, img_og, MIN_AREA, path)
    cv2.imwrite(f'json_output/{path}', img_pieces)

    return objs, len(colors)

def run_folder(folder):
    res = {
        "results": []
    }

    dest = "output"
    if folder == "easy_samples":
        dest = "easy_output"

    img_folder = os.listdir(folder)

    for path in img_folder:
        objs, num_colors = run(path)

        res.get("results").append({
            "file_name": path,
            "num_colors": num_colors,
            "num_detections": len(objs),
            "detected_objects": objs
        })

    with open(f'{dest}/results.json', 'w') as f:
        json.dump(res, f, indent=4)


def run_json(json_path):
    res = {
        "results": []
    }

    with open(json_path) as f:
        data = json.load(f)

    for path in data['image_files']:
        objs, num_colors = run(path)

        res.get("results").append({
            "file_name": path,
            "num_colors": num_colors,
            "num_detections": len(objs),
            "detected_objects": objs
        })

    with open('json_output/results.json', 'w') as f:
        json.dump(res, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LEGO Detector')

    parser.add_argument('--folder', help='Folder path')
    parser.add_argument('--json', help='Json path')

    args = parser.parse_args()

    if len(vars(args)) == 0:
        run()

    if args.folder:
        run_folder(args.folder)
    elif args.json:
        run_json(args.json)
