from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Conv2D
from detector import Detector
from dataset import create_dataset, load_dataset, encode_labels
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, MaxPool2D, Flatten
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from tqdm.keras import TqdmCallback
import numpy as np
import cv2


def compile_model(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy"):
    model = Sequential()

    # first layer - convolution
    model.add(Conv2D(32, 3, activation='relu', input_shape=(30, 30, 1)))

    # second layer - pooling
    model.add(MaxPool2D(pool_size=(2, 2)))

    # dropout
    model.add(Dropout(0.3))

    #
    model.add(Flatten())

    # output
    model.add(Dense(16, activation='softmax'))

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)
    return model


def main():
    detector = Detector(verbose=True)
    crop_list, crop_coord = detector.detect("./data/test2.jpg")
    # cv2.imshow("crop", crop_list[0])
    # cv2.waitKey(0)
    # create_dataset("./dataset/")

    X, y = load_dataset("./dataset/results/")
    X = np.asarray(X).astype('float32')
    y = encode_labels(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = compile_model()
    model.fit(X_train, y_train, epochs=1)
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    for crop in crop_list:
        image = np.array(crop)
        image = image.astype('float32')
        image /= 255
        print(model.predict(image))


if __name__ == '__main__':
    main()
