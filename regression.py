# Christoffer

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import csv


class MultiLinear():
    '''Represents one (or more) Regression models, based on 
    scikitlearn's linear model. Utilizes Bayesian Ridge regression 
    (part of scikitlearn module).
    
    '''

    def train_linear_model(self, train_features):
        '''Traine (Ridge) regression models with features (x) and 
        complexities (y) based on features of target words from the 
        CompLex corpus. 

        Parameters:

            train_features (FeatureMatrix-object)
                Object contains an array/matrix of entries with features 
                and corresponding lexical complexities per entry, based 
                ontraining data and manual complexity annotation from the 
                CompLex corpus. 
        
        Returns:

            models (list)
                a collection of trained regression models.

        '''
        # Murathan: This (____make_versions) is great thinking but I am not sure this is how you would want to do this. Among other things (see below for the other things),
        # Murathan: this should be optional, most of the time I would not be interested in the performance of the subsets of the features I specified
        # Murathan: but want to see the end result directly.
        
        #train_matrices = self.__make_versions(train_features.matrix)  # Gets multiple versions of the full matrix, with different sets of features included
        train_matrix = train_features.matrix
        train_compl = train_features.complexities  # Murathan: minor thing, I think a more conventional variable name would be "labels"
        models = []  # Murathan: It is not straightforward to understand which model is at which index? One needs to check the __make_versions method to see in which order you
                    # Murathan: add models to this list. I think it would be better to make this variable a dictionary (e.g. model["handcrafted"]=handcrafted_model)

        # Iterate through feature matrix versions, train regression models
        # and append to models list.
        # for train_matrix in train_matrices:
        #     regr = linear_model.BayesianRidge(verbose=True) # Murathan: set verbose=True
        #     regr.fit(train_matrix, train_compl) # Murathan: Why is there no option to control the # of iterations?
        #     models.append(regr)
        # return models

        regr = linear_model.BayesianRidge(verbose=True) # Murathan: set verbose=True
        regr.fit(train_matrix, train_compl) # Murathan: Why is there no option to control the # of iterations?
        return regr
        #models.append(regr)

    # Murathan: This method could have been static (a stand alone method) as its only connection to this class is the .__make_version function.
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
        # test_matrices = self.__make_versions(test_features.matrix)  # Gets multiple versions of the full matrix, with different sets of features included
        test_matrix = test_features.matrix
        test_compl = test_features.complexities

        # Names of versions of feature matrices.
        # feature_versions = ['all', 'handcrafted', 'clusters + outliers + embeddings',
        #                     'handcrafted + clusters + outliers',
        #                     'handcrafted + clusters + outliers + 50 dimensions']
        # Murathan: Great! you compute lots of different metrics. You could also print the feature set with the best performance on a certain metric separately at the end.
        # Create tuples of statistic measure and corresponding name 
        # to iterate and print results with.
        stat_functions = ((pearsonr, 'pearson\'s r = '),
                          (spearmanr, 'spearman\'s rho = '),
                          (mean_absolute_error, 'mae = '),
                          (mean_squared_error, 'mse = '),
                          (r2_score, 'r2 = '))

        # Iterate through matrices and corresponding trained regression models.
        # For each model, predict complexities and apply (5) statistic measures.
        # Print results.
        # for test_matrix, regr, features in zip(test_matrices, regr_models,
        #                                        feature_versions):
        #     compl_pred = regr.predict(test_matrix)
        #     print('\nFeatures: ', features)
        #     for stat, statname in stat_functions:
        #         result = stat(test_compl, compl_pred)
        #         print(statname, result)
        trial_ids = test_features.ids
        compl_pred = predict(test_matrix)
        self.__write_csv('trial', test_ids, compl_pred)
        for stat, statname in stat_functions:
            result = stat(test_compl, compl_pred)
            print(statname, result)

    # Murathan: This function asssumes that *all* features exist in the matrix. What will happen if I initialize the feature matrix with only "handcrafted_senses" in the main.py?
    # Murathan: Then, each row of the matrix will have only one feature (hence, will be of len 1), and your hardcoded indices below would not mean anything.
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
        new_matrices = [matrix, handcrafted, embeddings, handcrafted_senses,
                        handcrafted_senses_50_emb]

        return new_matrices

    # def cross_validate(self, feature_matrix):
    #     feat_ind_dict = {}
    #     feat_ind_list = []
    #     for feature, n in zip(feature_matrix, range(len(feature_matrix))):
    #         feat_ind_dict[n] = feature
    #         feat_ind_list.append(n)
    #     final_matrix = []
    #     ver_matrix = []
    #     loo = LeaveOneOut()
    #     for train, test in loo.split(feat_ind_list):
    #         this_version = []
    #         for index in train.tolist():
    #             this_version = this_version + feat_ind_dict[index]
    #         final_matrix.append(this_version)
    #     print(final_matrix)

        def write_results(self, regr, test_features):

            test_matrix = test_features.matrix
            test_ids = test_features.ids
            compl_pred = regr.predict(test_matrix)
            self.__write_csv('test', test_ids, compl_pred)
            
            

        def __write_csv(self, mode, ids, pred):
            with open('results' + mode +'.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for idn, prediction in zip(test_ids, compl_pred):
                    writer.writerow([idn, prediction])

