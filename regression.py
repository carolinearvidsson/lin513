# Christoffer

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

class MultiLinear():
    '''Represents one (or more) Regression models, based on 
    scikitlearn's linear model. Utilizes Bayesian Ridge regression 
    (part of scikitlearn module)
    
    '''

    def train_linear_model(self, train_features):
        '''Return (Ridge) regressian models trained with features (x) 
        and complexities (y) based from the CompLex corpus. 

        Parameters:

            train_features (FeatureMatrix-object)
                Object contains an array of entries with features and 
                corresponding lexical complexities per entry, based on
                training data from the CompLex corpus. 
        
        Returns:

            models (list)
                a collection of trained regression models.

        '''

        train_matrices = self.__make_versions(train_features.matrix)
        train_compl = train_features.complexities
        models = []
        # Iterate through feature matrix versions, train regression models 
        # and append to models list.
        for train_matrix in train_matrices: 
            regr = linear_model.BayesianRidge()
            regr.fit(train_matrix, train_compl)
            models.append(regr)
        return models

    def predict(self, regr_models, test_features):
        '''Method predicts complexity values from features in test data, 
        using previously trained regression models. Prints correlation 
        and error measures of comparison with manually annotated complexitiees
        (same as SemEval2021 Baseline (https://github.com/MMU-TDMLab/CompLex),
        5 statistic measures). 
        
        Parameters:

            regr_models (list)
                Elements are regression models, trained on different 
                combinations of features.
            
            test_features (FeatureMatrix-object)
                Object ontains an array of entries with features and 
                corresponding lexical complexities per entry, based on
                test data from the ComPlex corpus. 
        '''
        test_matrices = self.__make_versions(test_features.matrix)
        test_compl = test_features.complexities
        
        # Names of versions of feature matrices.
        feature_versions = ['all', 'handcrafted', 'clusters + outliers + embeddings', 
                            'handcrafted + clusters + outliers', 
                            'handcrafted + clusters + outliers + 50 dimensions']
        
        # Create tuples of statistic measure and corresponding string 
        # to iterate and print results with.
        stat_functions = ((pearsonr, 'pearson\'s r = '), (spearmanr, 'spearman\'s rho = '),
                          (mean_absolute_error, 'mae = '), (mean_squared_error, 'mse = '), 
                          (r2_score, 'r2 = '))
       
       # Iterate through matrices and corresponding trained regression models. 
       # For each model, predict complexities and apply (5) statistic measures. 
       # Print results.
        for test_matrix, regr, features in zip(test_matrices, regr_models, \
                                                feature_versions):
            compl_pred = regr.predict(test_matrix)
            print('\nFeatures: ', features)
            for stat, statname in stat_functions:
                result = stat(test_compl, compl_pred)
                print(statname, result)
        
    def __make_versions(self, matrix):
        '''Creates versions of the original feature matrix. Excludes 
        certain features to test which ones may be more effective. 
        Handcrafted features are defined as those who can not be derived
        from BERT-embeddings. Senses are defined as features using the
        embeddings, but do not include the vector dimensions themselves. 
        
        Parameters:
            
            matrix (array)
                a feature array where every entry has 783 features.  

        Returns:

            list of original and modified feature matrices, excluding 
            certain features.  
        '''
        
        handcrafted = [row[:14] for row in matrix]
        embeddings = [row[14:] for row in matrix]
        handcrafted_senses = [row[:16] for row in matrix]
        handcrafted_senses_50_emb = [row[:16] + row[-50:] for row in matrix]

        return [matrix, handcrafted, embeddings, handcrafted_senses, handcrafted_senses_50_emb]

        


