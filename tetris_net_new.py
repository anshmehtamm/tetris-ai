from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
import tensorflow as tf


class TetrisQNet:

    def __init__(self):
        self.conv_model = self.create_conv_model()

    def create_conv_model(self):
        model = Sequential()
        # conv net, input shape is 177x43x1
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(177,43,1)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

    def create_

