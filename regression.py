# Christoffer

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

class MultiLinear():

    def train_linear_model(self, train_features):

        train_matrices = self.__make_versions(train_features.matrix)
        train_compl = train_features.complexities
        print(len(train_compl))
        models = []
        for train_matrix in train_matrices: #!!!!!! ändra tillbaka till train_matrices
            regr = linear_model.BayesianRidge()#.BayesianRidge() # Använda en annan? linear_model.Ridge(alpha=0.5)
            regr.fit(train_matrix, train_compl)
            models.append(regr)
        return models

    def predict(self, regr_models, test_features):
        test_matrices = self.__make_versions(test_features.matrix)
        test_compl = test_features.complexities
        feature_versions = ['all', 'handcrafted', 'clusters + outliers + embeddings', 
                            'handcrafted + clusters + outliers', 
                            'handcrafted + clusters + outliers + 50 dimensions']
        
        stat_functions = ((pearsonr, 'pearson\'s r = '), (spearmanr, 'spearman\'s rho = '),
                          (mean_absolute_error, 'mae = '), (mean_squared_error, 'mse = '), 
                          (r2_score, 'r2 = '))
       
        for test_matrix, regr, features in zip(test_matrices, regr_models, feature_versions):
            print('\nFeatures: ', features)
            compl_pred = regr.predict(test_matrix)
            for stat, statname in stat_functions:
                result = stat(test_compl, compl_pred)
                print(statname, result)
        
    def __make_versions(self, matrix):
        
        handcraft = [row[:14] for row in matrix]
        embeddings = [row[14:] for row in matrix]
        handcraft_senses = [row[:16] for row in matrix]
        handcrafted_senses_50_emb = [row[:16] + row[-50:] for row in matrix]

        return [matrix, handcraft, embeddings, handcraft_senses, handcrafted_senses_50_emb]

        


