from sklearn import ensemble

from . import DigitClassificationInterface


HYPERPARAMETERS = {}


class RandomForest(DigitClassificationInterface):
    def __init__(self):
        self.model = ensemble.RandomForestClassifier(**HYPERPARAMETERS)
        
    def predict(self, sample):
        return self.model.predict([sample])[0]
