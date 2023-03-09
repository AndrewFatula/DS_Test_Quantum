import pandas as pd
import pickle

class Model:
    def fit(self, x, y):
        pass
    def predict(self, x):
        return x**2 + 0.5
    
    
train = pd.read_csv('data/train.csv')
regressor = Model()
regressor.fit(train.drop('target', axis=1), train['target'])

with open('model/model.pickle', 'wb') as w_file:
    pickle.dump(regressor, w_file)
