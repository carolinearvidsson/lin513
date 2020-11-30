from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from features import FeatureMatrix

import pickle

class MultiLinear():
        
    def get_value(self, train_feature_matrix, test_feature_matrix):
        
        train_raw = pickle.load(open(train_feature_matrix, 'rb'))
        train = [[],train_raw[1]]
        for i, entry in enumerate(train_raw[0]):
            emb = entry[5]
            try:
                emba = emb.tolist()
                entry = entry[:5] + emba[50:101]
                train[0].append(entry)
            except:
                train[1].pop(i)
            
            
            #print(entry)
            
            #print(entry)
            #break
            
        test_raw = pickle.load(open(test_feature_matrix, 'rb'))
        test = [[], test_raw[1]]
        for i, entry in enumerate(test_raw[0]):
            emb = entry[5]
            try:
                emba = emb.tolist()
                entry = entry[:5] + emba[50:101]
                test[0].append(entry)
            except: 
                test[1].pop(i)
        #print(train,test)
        print(self.__predict(self.__train_linear_model(train), test))

    def __train_linear_model(self, train_matrix):

        print(len(train_matrix[0]), len(train_matrix[1]))
        regr = linear_model.BayesianRidge()
        regr.fit(train_matrix[0], train_matrix[1])
        return regr

    def __predict(self, regr, test_matrix):
        complexities_predictions = regr.predict(test_matrix[0])
        mae = mean_absolute_error(test_matrix[1], complexities_predictions)
        return mae


