from keras.layers import Dense, Dropout, MaxPool2D, Flatten
from keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D


class ModelHelper:
    """
    Implements functions needed for building basic CN with Keras
    Parameters:
        optimizer: Optimizer used in model, defualts to "Adam"
        loss: Loss function used in model training and validation, defaults to "categorical_crossentropy"
        metrics: Metrics used in model evaluation, defaults to "accuracy"
    Methods:
        compile_model(): returns compiled model with default parameters and four layers
        predict(): gives prediction of given image
    """

    def __init__(self, optimizer="adam", loss="categorical_crossentropy", metrics="accuracy"):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.model = None

    def compile_model(self, conv_activation="relu", input_shape=(45, 45, 1), pool_size=(2, 2), dropout=0.3):
        model = Sequential()

        # first layer - convolution
        model.add(Conv2D(32, 3, activation=conv_activation, input_shape=input_shape))

        # second layer - pooling
        model.add(MaxPool2D(pool_size=pool_size))

        # dropout
        model.add(Dropout(dropout))
        model.add(Flatten())

        # output
        model.add(Dense(16, activation='softmax'))

        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics)

        return model