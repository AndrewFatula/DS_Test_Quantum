import numpy as np
import tensorflow as tf

from . import DigitClassificationInterface


class NeuralNetwork(DigitClassificationInterface):
    def __init__(self):
        num_classes = 10
        self.model = tf.keras.Sequential([])
        self.model.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 5),activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dense(num_classes))
                       
    def predict(self, sample):
        res = self.model(sample[None, :,:,:])
        return np.argmax(res, axis=1)[0]
