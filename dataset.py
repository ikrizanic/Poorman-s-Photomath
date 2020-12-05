import numpy as np
import os
import cv2
import pickle

IMG_WIDTH = 45
IMG_HEIGHT = 45


def create_dataset(img_folder):
    for folder in os.listdir(img_folder):
        img_data = []
        if folder == "results":
            continue
        print(folder)
        for file in os.listdir(os.path.join(img_folder, folder)):
            image_path = os.path.join(img_folder, folder, file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            image = cv2.bitwise_not(image)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data.append([image, folder])
        with open("./dataset/results/" + folder + ".pickle", "wb") as pickle_file:
            pickle.dump(img_data, pickle_file)


def load_dataset(dataset_folder):
    dataset = []
    for file in os.listdir(dataset_folder):
        with open(dataset_folder + file, "rb") as pickle_file:
            data = pickle.load(pickle_file)
            dataset.extend(data)

    dataset = np.array(dataset)
    X, y = dataset[:, 0], dataset[:, 1]

    Xn = np.zeros(shape=(len(X), len(X[0]), len(X[0][0]), 1))
    for i in range(len(X)):
        Xn[i] = np.expand_dims(X[i], axis=(0, 3))
    return Xn, y


def encode_labels(labels):
    encoder = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "+": 10,
        "-": 11,
        "div": 12,
        "times": 13,
        "(": 14,
        ")": 15
    }
    res = np.zeros(len(labels))
    for i in range(len(labels)):
        res[i] = encoder.get(labels[i])
    return res


def decode_predictions(predictions):
    result = ""
    decoder = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
        10: "+",
        11: "-",
        12: "/",
        13: "*",
        14: "(",
        15: ")"
    }
    for p in predictions:
        result += decoder.get(p)
    return result
