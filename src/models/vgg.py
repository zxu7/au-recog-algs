import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Input, BatchNormalization, Flatten, Dropout, Dense
from keras.layers.core import Dense
from keras.optimizers import SGD, Adam
from keras import backend as K


def smallvgg(w, h, depth=3, classes=14):
    inp = Input((h, w, depth), name='input')
    x = Conv2D(32, (3, 3), padding="same", activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPool2D((3,3))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding="same", activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding="same", activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(classes, activation='sigmoid')(x)
    model = Model(inp, x)
    model.compile(Adam(1e-3), loss=K.binary_crossentropy, metrics=[keras.metrics.binary_accuracy,])
    return model

