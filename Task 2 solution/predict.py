import pandas as pd
import pickle

class Model:
    def fit(self, x, y):
        pass
    def predict(self, x):
        return x**2 + 0.5
    
with open('model/model.pickle', 'rb') as r_file:
    regressor = pickle.load(r_file)

    
test = pd.read_csv('data/hidden_test.csv')
test_filtered = test[['6']]

preds = regressor.predict(test_filtered)

test['target']=preds
test.to_csv('output/hidden_predicted.csv', index=False)
