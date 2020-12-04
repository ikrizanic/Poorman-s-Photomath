import cv2
from imutils import contours
import numpy as np


class Detector:
    def __init__(self, verbose=False, resize=True, width=400, height=400):
        self.verbose = verbose
        self.resize = resize
        self.width = width
        self.height = height

    def detect(self, image_path):
        # read image from path
        image = cv2.imread(image_path)
        if self.resize:
            height, width, _ = image.shape
            image = cv2.resize(image, (min(self.width, width), min(self.height, height)),
                               interpolation=cv2.INTER_AREA)

        # turn image into gray-scale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # adaptive thresholding
        image_proc = cv2.adaptiveThreshold(image_gray, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 20)

        # image dilation
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        image_proc = cv2.dilate(image_proc, rect_kernel, iterations=1)

        # noise removal with opening(erode + dilate)
        # noise_kernel = np.ones((2, 2), np.uint8)
        # image_proc = cv2.morphologyEx(image_proc, cv2.MORPH_OPEN, noise_kernel)

        # find contours
        cnts, hierarchy = cv2.findContours(image_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts, _ = contours.sort_contours(cnts, method="left-to-right")

        # save contours anc coordinates in lists
        crop_list = list()
        crop_coord = list()
        for c in cnts:
            if cv2.contourArea(c) > 40:
                x, y, w, h = cv2.boundingRect(c)
                crop = 255 - image[y:y + h, x:x + w]
                crop_list.append(crop)
                crop_coord.append([x, y, w, h])
                if self.verbose:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)

        if self.verbose:
            cv2.imshow("Threshold", image_proc)
            cv2.imshow("Image", image)
            cv2.waitKey(0)
        return np.array(crop_list), np.array(crop_coord)
