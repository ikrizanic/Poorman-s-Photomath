from src.helpers import load_dataset, encode_labels, create_dataset, decode_predictions, predict_images, solve
from src.model import ModelHelper
import numpy as np
import cv2
from keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
from detector import Detector
from src.solver import parse_and_solve


def main():
    detector = Detector(verbose=False)
    crop_list, crop_coord = detector.detect("./test_images/test1.png")
    # crop_list, crop_coord = detector.detect("./test_images/test2.jpg")

    # CREATE LOAD DATASET
    # create_dataset("./dataset/")
    # X, y = load_dataset("./dataset/results/")
    #
    # X = np.asarray(X).astype('float32')
    # y = encode_labels(y)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    #
    # # one hot encoding
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    #
    # # build Keras model
    # mh = ModelHelper()
    # model = mh.compile_model()
    # model.fit(X_train, y_train, epochs=1)
    # val_loss, val_acc = model.evaluate(X_test, y_test)

    # SAVE LOAD MODEL
    # model.save("./model")
    model = load_model("./model")

    task = predict_images(model, crop_list, verbose=True)
    print(task)
    solution = parse_and_solve(task)
    if solution is not None:
        print("Uspješno rješavanje izraza: ", task)
        print("Rješenje je: ", solution)
    else:
        print("Pročitani izraz glasi: ", task)
        print("Nažalost, izraz je pogrešno pročitan ili pogrešno zadan.")


if __name__ == '__main__':
    main()
