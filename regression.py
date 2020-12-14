from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

class MultiLinear():
    #hej    
    # def get_value(self, train_feature_matrix, test_feature_matrix):
        
    #     train_raw = pickle.load(open(train_feature_matrix, 'rb'))
    #     train_matrix = train_raw.matrix
    #     train_compl = train_raw.complexities
    #     # train = [[],train_compl]
    #     # for i, entry in enumerate(train_matrix):
    #     #     try:
    #     #         entry = entry[1:]
    #     #         train[0].append(entry)
    #     #     except:
    #     #         train[1].pop(i)
            
            
    #         #print(entry)
            
    #         #print(entry)
    #         #break
            
    #     test_raw = pickle.load(open(test_feature_matrix, 'rb'))
    #     test_matrix = test_raw.matrix
    #     test_compl = test_raw.complexities
    #     # test = [[], test_compl]
    #     # for i, entry in enumerate(test_matrix):
    #     #     try:
    #     #         entry = entry[1:]
    #     #         test[0].append(entry)
    #     #     except: 
    #     #         test[1].pop(i)
    #     #print(train,test)
    #     print(self.__predict(self.__train_linear_model(train_matrix, train_compl), test_matrix, test_compl))

    def train_linear_model(self, train_features):

        #train_matrices = self.__make_versions(train_features.matrix)
        train_matrix = train_features.matrix
        train_compl = train_features.complexities
        print(len(train_compl))
        models = []
        regr = linear_model.BayesianRidge() # Använda en annan? linear_model.Ridge(alpha=0.5)
        regr.fit(train_matrix, train_compl)
        # models.append(regr)
        return regr
        # for train_matrix in train_matrices:
        #     regr = linear_model.BayesianRidge() # Använda en annan? linear_model.Ridge(alpha=0.5)
        #     regr.fit(train_matrix, train_compl)
        #     models.append(regr)
        # return models

    def predict(self, regr_models, test_features):
        # test_matrices = self.__make_versions(test_features.matrix)
        test_matrix = test_features.matrix
        test_compl = test_features.complexities
        # feature_versions = ['all', 'handcrafted', 'clusters + outliers + embeddings', 
        #                     'handcrafted + clusters + outliers', 
        #                     'handcrafted + clusters + outliers + 50 dimensions']
        # stat_functions = ((pearsonr, 'pearson\'s r = '), (spearmanr, 'spearman\'s rho = '),
        #                  (mean_absolute_error, 'mae = '), (mean_squared_error, 'mse = '), 
        #                  (r2_score, 'r2 = '))
        # for test_matrix, regr, features in zip(test_matrices, regr_models, feature_versions):
        #     print('Features: ', features)
        #     compl_pred = regr.predict(test_matrix)
        #     for stat, statname in stat_functions:
        #         result = stat(test_compl, compl_pred)
        #         print(statname, result)
        r_value = pearsonr(test_compl, compl_pred)
        rho = spearmanr(test_compl, compl_pred)
        mae = mean_absolute_error(test_compl, compl_pred)
        mse = mean_squared_error(test_compl, compl_pred)
        r_2 = r2_score(test_compl, compl_pred)
        print('Features: ', features, '\nPearson\'s r = ', r_value, '\nSpearman\'s rho = ', rho,
            '\nMAE = ', mae, '\nMSE = ', mse, '\nr2 = ', r_2 )
        
    def __make_versions(self, matrix):
        matrices = [matrix]
        matrices.append([row[:10] for row in matrix])
        matrices.append([row[10:] for row in matrix])
        matrices.append([row[:12] for row in matrix])
        matrices.append([row[:12] + row[-50:] for row in matrix])

        return [matrices]
        


