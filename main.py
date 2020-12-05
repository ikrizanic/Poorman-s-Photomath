import numpy as np
import cv2
from keras.layers import Dense, Dropout, MaxPool2D, Flatten
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Conv2D
from dataset import load_dataset, encode_labels, create_dataset, decode_predictions
from detector import Detector


def compile_model(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy"):
    model = Sequential()

    # first layer - convolution
    model.add(Conv2D(32, 3, activation='relu', input_shape=(45, 45, 1)))

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
    crop_list, crop_coord = detector.detect("./data/pc_test.png")
    # crop_list, crop_coord = detector.detect("./data/test3.jpg")
    # cv2.imshow("crop", crop_list[0])
    # cv2.waitKey(0)
    # create_dataset("./dataset/")

    X, y = load_dataset("./dataset/results/")
    X = np.asarray(X).astype('float32')
    y = encode_labels(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # model = compile_model()
    # model.fit(X_train, y_train, epochs=1)
    # val_loss, val_acc = model.evaluate(X_test, y_test)
    # print(val_loss)
    # print(val_acc)
    # #
    # model.save("./models")

    model = load_model("./models/")
    # for i in range(10):
    #     image = X[np.random.randint(0, 150000)]
    #     cv2.imshow("test", image)
    #     cv2.waitKey(0)
    #     print(np.argmax(model.predict(np.expand_dims(image, axis=0))))
    #
    predictions = []
    for i, crop in enumerate(crop_list):
        img = np.array(crop)
        img = img.astype('float32')
        img /= 255
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        p = np.argmax(model.predict(np.expand_dims(img, axis=(0, 3))))
        predictions.append(p)
        print(p)
    d = decode_predictions(predictions)
    print("Rezultat traženog izraza: \n", d, " = ", eval(d))
    cv2.waitKey(0)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
