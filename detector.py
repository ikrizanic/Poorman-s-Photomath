import cv2
from imutils import contours
import numpy as np


class Detector:
    """Class for detection and retrieval of contours on images
    Parameters:
        verbose (boolean): If true shows image and contours on display
        resize (boolean): If true image will be resized before contours search
        width (int): If resize is true, image width will be min(original_width, width)
        height (int): If resize is true, image height will be min(original_height, height)

    Returns:
        crop_list (np.array): numpy array of cropped images
        crop_coord (np.array): numpy array of cropped images coordinates

    Methods:
        detect(image_path): Method that searches for contours and returns them as stated
    """

    def __init__(self, verbose=False, resize=True, width=800, height=400):
        self.verbose = verbose
        self.resize = resize
        self.width = width
        self.height = height

    def detect(self, image_path):
        # read image from path
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        if width > height:
            image = image_resize(image, width=min(width, self.width))
        else:
            image = image_resize(image, height=min(height, self.height))
        height, width, _ = image.shape

        # turn image into gray-scale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # adaptive thresholding
        image_proc = cv2.adaptiveThreshold(image_gray, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 20)

        # image dilation
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(max(height, width)/100), int(max(height, width)/100)))
        image_dil = cv2.dilate(image_proc, rect_kernel, iterations=1)

        # find contours
        cnts, hierarchy = cv2.findContours(image_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts, _ = contours.sort_contours(cnts, method="left-to-right")

        # save contours anc coordinates in lists
        crop_list = list()
        crop_coord = list()
        erode_kernel = np.ones(shape=(2, 2), dtype=np.uint8)

        coef = 0.003
        num = 0
        while num < len(cnts) / 1.2:
            num = 0
            coef /= 2
            for c in cnts:
                if cv2.contourArea(c) > coef * width * height:
                    num += 1

        for c in cnts:
            if cv2.contourArea(c) > coef * width * height:
                x, y, w, h = cv2.boundingRect(c)
                crop = image_proc[y:y + h, x:x + w]
                crop = image_fill(crop)
                crop = cv2.dilate(crop, erode_kernel, iterations=1)
                crop = cv2.erode(crop, erode_kernel, iterations=1)
                crop_list.append(crop)
                crop_coord.append([x, y, w, h])
                if self.verbose:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)

        if self.verbose:
            cv2.imshow("Threshold", image_proc)
            cv2.imshow("Image", image)
            cv2.waitKey(0)
        return np.array(crop_list), np.array(crop_coord)


def image_fill(image):
    i_w, i_h = image.shape
    if i_w > i_h:
        image = image_resize(image, height=45)
    else:
        image = image_resize(image, width=45)

    s = max(image.shape[0:2])
    f = np.zeros((s, s), np.uint8)
    # Getting the centering position
    ax, ay = (s - image.shape[1]) // 2, (s - image.shape[0]) // 2

    # Pasting the 'image' in a centering position
    f[ay:image.shape[0] + ay, ax:ax + image.shape[1]] = image
    return f


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
