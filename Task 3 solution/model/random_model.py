import numpy as np

from . import DigitClassificationInterface

class RandomModel(DigitClassificationInterface):
    def predict(self, sample):
        return np.random.randint(0,10)