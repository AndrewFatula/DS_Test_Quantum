from model.random_forest import RandomForest
from model.neural_network import NeuralNetwork, np
from model.random_model import RandomModel

class DigitClassiier:
    def __init__(self, model, hyperparameters={}):
        if model=='rf':
            self.model_type='rf'
            self.model=RandomForest(hyperparameters)
        elif model=='cnn':
            self.model_type='rf'
            self.model=NeuralNetwork()
        elif model=='rand':
            self.model_type='rand'
            self.model=RandomModel()
        else:
            raise Exception('Wrong model type specified, should be one of ["rf", "cnn", "rand"] ')
        
    def fit(self, x, y):
        raise NotImplementedError
        
    def predict(self, sample: np.array):
        if not isinstance(sample, np.ndarray):
            raise TypeError('input type should be numpy.ndarray')
        if not len(sample.shape)==3:
            raise Exception('input array should be 3-dimensional')
        if not (sample.shape[0]==sample.shape[1]==28 and shape.[2]==1):
            raise Exception('input array shape should be (28,28,1)')        
        if self.model_type=='rf':
            sample = np.ravel(sample)
            return self.model.predict(sample)
        elif self.model_type=='cnn':
            sample = sample
            return self.model.predict(sample)
        elif self.model_type=='rand':
            sample = sample[9:-9, 9:-9]
            return self.model.predict(sample)
        
