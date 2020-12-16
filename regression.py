# Christoffer

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

class MultiLinear():
    '''Regression model, based on scikitlearn's linear model.
    Utilizes Bayesian Ridge regression (part of scikitlearn module)
    
    
    '''

    def train_linear_model(self, train_features):
        '''Return (Ridge) regressian models trained with features (x) 
        and complexities (y) based from the CompLex corpus. 

        Parameters:

            train_features (FeatureMatrix-object)
                Object contains an array of entries with features and 
                corresponding lexical complexities per entry. 
        '''

        train_matrices = self.__make_versions(train_features.matrix)
        train_compl = train_features.complexities
        models = []
        for train_matrix in train_matrices: #!!!!!! ändra tillbaka till train_matrices
            regr = linear_model.BayesianRidge()#.BayesianRidge() # Använda en annan? linear_model.Ridge(alpha=0.5)
            regr.fit(train_matrix, train_compl)
            models.append(regr)
        return models

    def predict(self, regr_models, test_features):
        '''Print results different statistic measures (same as SemEval2021 
        baseline measures) based on testing predicted complexities vs. 
        manually annotated of CompLex corpus. Tests multiple (5) previously 
        trained models. 
        
        Parameters:

            regr_models (list)
                Elements are regression models, trained on different 
                combinations of features.
            
            test_features (FeatureMatrix-object)
                Object ontains an array of entries with features and 
                corresponding lexical complexities per entry. 
        '''
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
        '''Return modified feature matrices to be able to train and test
        multiple models with different feature combinations.
        
        Parameters:
            
            matrix (array)
                a feature array where every entry has 783 features.    
        '''
        
        handcraft = [row[:14] for row in matrix]
        embeddings = [row[14:] for row in matrix]
        handcraft_senses = [row[:16] for row in matrix]
        handcrafted_senses_50_emb = [row[:16] + row[-50:] for row in matrix]

        return [matrix, handcraft, embeddings, handcraft_senses, handcrafted_senses_50_emb]

        


