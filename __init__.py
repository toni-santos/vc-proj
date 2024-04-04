import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import time

def overlapOverThreshold(rect1, rect2, area_threshold):
    # Two rects overlap by more than 60% of the area of either rectangle?
    x1_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
    y1_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))
    overlap_area = x1_overlap * y1_overlap

    area_rect1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area_rect2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

    total_area = area_rect1 + area_rect2 - overlap_area

    overlap_percentage = max((overlap_area / area_rect1) * 100, (overlap_area / area_rect2) * 100)
    print(overlap_percentage)

    return overlap_percentage > area_threshold

def overlapUnion(rect1, rect2, threshold):
    if (rect1[0] - rect2[2] >= threshold or 
        rect2[0] - rect1[2] >= threshold or
        rect1[1] - rect2[3] >= threshold or 
        rect2[1] - rect1[3] >= threshold):
        return None

    print("Merging: ", rect1, rect2)

    x1 = min(rect1[0], rect2[0])
    y1 = min(rect1[1], rect2[1])
    x2 = max(rect1[2], rect2[2])
    y2 = max(rect1[3], rect2[3])
    

    # x y w h
    return [x1, y1, x2-x1, y2-y1]

def getColor(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
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

def preProcessing(img):
    print(f"Preprocessing...")
    start_time = time.time()

    # Denoising
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    img_canny = cv2.Canny(img, 50, 70)

    # Morphological operations
    img_canny = cv2.morphologyEx(img_canny, cv2.MORPH_DILATE, kernel, iterations=1)
    img_canny = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Thresholding
    thresh = cv2.threshold(img_canny, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    print(f"Canny edge detection done in {time.time() - start_time} seconds")

    return contours, img_canny

def findPieces(contours, img, img_og, MIN_AREA, gc = False):
    print("Finding pieces...")
    start_time = time.time()
    copy_og = img_og.copy()
    objs = []
    colors = []
    overlap = True
    draw = True
    MIN_OVERLAP_AREA = MIN_AREA * 0.5
    pop_list = []
    rects = [cv2.boundingRect(cntr) for cntr in contours]
    while overlap or not draw:
        overlap = False
        for idx, rect in enumerate(list(rects)):
            draw = True
            if (idx == 0):
                rects_copy = rects
                rects = []
            print("Current rect: ", idx)
            x,y,w,h = rect
            area = w * h
            if area < MIN_OVERLAP_AREA:
                continue
            print(x, y, w, h)
            MAX_DIST = min(img.shape[0], img.shape[1]) * 0.003
            print("max_dist: ", MAX_DIST)

            for n_idx, n_rect in enumerate(list(rects_copy)):
                print("Checking with: ", n_idx)
                x1,y1,w1,h1 = n_rect
                n_area = w1*h1
                # Check if the current contour is inside another contour and vice-versa
                if (n_idx != idx) and (x >= x1 and y >= y1 and x+w <= x1+w1 and y+h <= y1+h1):
                    print("inside")
                    draw = False
                    break
                # Check if the current contour is inside/overlapping another contour
                elif (n_idx != idx) and (overlapOverThreshold([x, y, x+w, y+h], [x1, y1, x1+w1, y1+h1], 60)
                    ):
                    print("overlap")
                    overlap = True
                    x, y, w, h = overlapUnion([x, y, x+w, y+h], [x1, y1, x1+w1, y1+h1], MAX_DIST)
                    print("Joined rect size: ", x, y, w, h)
                elif (n_idx != idx) and (overlapUnion([x, y, x+w, y+h], [x1, y1, x1+w1, y1+h1], MAX_DIST)
                    and n_area < MIN_AREA):
                    print("overlap")
                    overlap = True
                    x, y, w, h = overlapUnion([x, y, x+w, y+h], [x1, y1, x1+w1, y1+h1], MAX_DIST)
                    print("Joined rect size: ", x, y, w, h)
            
            area = w * h
            if area < MIN_AREA:
                print("Area too small")
                continue
            if draw:
                rects.append([x, y, w, h])
            print(rects)
            rects = sorted(rects)
            rects = [rects[i] for i in range(len(rects)) if i == 0 or rects[i] != rects[i-1]]
    for rect in rects:
        x, y, w, h = rect
        print("size: ", w, img.shape[1])
        if (w/img.shape[1] > 0.8) or (h/img.shape[0] > 0.8):
            continue
        cv2.rectangle(copy_og, (x, y), (x+w, y+h), (0, 0, 255), 2)

        objs.append({
            "xmin": x,
            "ymin": y,
            "xmax": x + w,
            "ymax": y + h
        })

        # Calculate the main color of the piece
        cropped = img_og[y:y+h, x:x+w]
        piece_color = tuple(getColor(cropped))
        if not inColorThreshold(colors, piece_color,30):
            colors.append(piece_color)

    print(f"Found {len(objs)} pieces and {len(colors)} colors in {time.time() - start_time} seconds")

    return colors, objs, copy_og

def run(path):
    print(f"Running on {path}")

    img_path = f"samples/{path}"
    canny_lower = 50
    canny_upper = 80
    threshold = 100
    maxval = 255
    MAX_PIECES = 30

    # Read image
    img = cv2.imread(img_path)
    # img = cv2.resize(img, (0, 0), fx=0.70, fy=0.70)
    img_og = img.copy()
    MIN_AREA = img.shape[0] * img.shape[1] * 0.006

    img = preProcessing(img)

    contours, img_canny = canny(img, canny_lower, canny_upper, threshold, maxval)
    cv2.imwrite(f'json_output/canny_{path}', img_canny)

    colors, objs, img_pieces = findPieces(contours, img, img_og, MIN_AREA)
    cv2.imwrite(f'json_output/{path}', img_pieces)

    if len(objs) > MAX_PIECES:
        print("Found too many pieces, running GrabCut...")
        start_time = time.time()
        img = img_og.copy()
        mask = np.zeros(img.shape[:2],np.uint8)
        rect = (1,1,img.shape[1],img.shape[0])
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        mask, bgdModel, fgdModel = cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask[:,:,np.newaxis]
        cv2.imwrite(f"json_output/grabcut_{path}", img)
        print(f"GrabCut done in {time.time() - start_time} seconds")

        img = preProcessing(img)

        contours, img_canny = canny(img, canny_lower, canny_upper, threshold, maxval)
        cv2.imwrite(f'json_output/canny_{path}', img_canny)

        new_colors, new_objs, img_pieces = findPieces(contours, img, img_og, MIN_AREA, gc=True)

        if len(new_objs) < len(objs):
            print("GrabCut pieces result is better")
            objs = new_objs
            cv2.imwrite(f'json_output/{path}', img_pieces)
        else:
            print("Original pieces result is better")

        # TODO: consider adding a similar heuristic for colors
        if len(new_colors) > len(colors):
            print("GrabCut colors result is better")
            colors = new_colors
        else:
            print("Original colors result is better")

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
