from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from features import FeatureMatrix

import pickle

class MultiLinear():
        
    def get_value(self, train_feature_matrix, test_feature_matrix):
        
        train_raw = pickle.load(open(train_feature_matrix, 'rb'))
        train_matrix = train_raw.matrix
        train_compl = train_raw.complexities
        train = [[],train_compl]
        for i, entry in enumerate(train_matrix):
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
        test_matrix = test_raw.matrix
        test_compl = test_raw.complexities
        test = [[], test_compl]
        for i, entry in enumerate(test_matrix):
            emb = entry[5]
            try:
                emba = emb.tolist()
                entry = entry[:5] + emba[50:101]
                test[0].append(entry)
            except: 
                test[1].pop(i)
        #print(train,test)
        print(self.__predict(self.__train_linear_model(train[0], train[1]), test[0], test[1]))

    def __train_linear_model(self, train_matrix, train_compl):

        print(len(train_matrix), len(train_compl))
        regr = linear_model.BayesianRidge()
        regr.fit(train_matrix, train_compl)
        return regr

    def __predict(self, regr, test_matrix, test_compl):
        complexities_predictions = regr.predict(test_matrix)
        mae = mean_absolute_error(test_compl, complexities_predictions)
        return mae


