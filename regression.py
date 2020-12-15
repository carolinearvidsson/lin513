# Christoffer

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

class MultiLinear():

    def train_linear_model(self, train_features):

        # train_matrices = self.__make_versions(train_features.matrix)
        train_matrices_2 = self.__make_versions_2(train_features.matrix)
        train_compl = train_features.complexities
        print(len(train_compl))
        models = []
        for train_matrix in train_matrices_2: #!!!!!! ändra tillbaka till train_matrices
            regr = linear_model.Ridge(alpha=0.5)#.BayesianRidge() # Använda en annan? linear_model.Ridge(alpha=0.5)
            regr.fit(train_matrix, train_compl)
            models.append(regr)
        return models

    def predict(self, regr_models, test_features):
        # test_matrices = self.__make_versions(test_features.matrix)
        test_matrices_2 = self.__make_versions_2(test_features.matrix)
        test_compl = test_features.complexities
        # feature_versions = ['all', 'handcrafted', 'clusters + outliers + embeddings', 
                            # 'handcrafted + clusters + outliers', 
                            # 'handcrafted + clusters + outliers + 50 dimensions']
        feature_versions_2 = ['ngram', 'word length', 'syll count', 'pos', 'all sen', 'lex sen', 'domain', 'freq', 'cluster', 'outlier' ]
        # stat_functions = ((pearsonr, 'pearson\'s r = '), (spearmanr, 'spearman\'s rho = '),
                        #  (mean_absolute_error, 'mae = '), (mean_squared_error, 'mse = '), 
                        #  (r2_score, 'r2 = '))
        for test_matrix, regr, features in zip(test_matrices_2, regr_models, feature_versions_2):
            print(features)
            compl_pred = regr.predict(test_matrix)
            mae = mean_absolute_error(test_compl, compl_pred)
            print(mae)
        # for test_matrix, regr, features in zip(test_matrices, regr_models, feature_versions):
        #     print('\nFeatures: ', features)
        #     compl_pred = regr.predict(test_matrix)
        #     for stat, statname in stat_functions:
        #         result = stat(test_compl, compl_pred)
        #         print(statname, result)
        
    def __make_versions(self, matrix):
        
        handcraft = [row[:13] for row in matrix]
        embeddings = [row[13:] for row in matrix]
        handcraft_simp_emb = [row[:15] for row in matrix]
        handcrafted_simp_50_emb = [row[:15] + row[-50:] for row in matrix]

        return [matrix, handcraft, embeddings, handcraft_simp_emb, handcrafted_simp_50_emb]
    
    def __make_versions_2(self, matrix):

        no_ngram = [row[3:15] for row in matrix]
        no_word_length = [row[:3] + row[4:15] for row in matrix]
        no_syll = [row[:4] + row[5:13] for row in matrix]
        no_pos = [row[:5] + row[9:15] for row in matrix]
        no_all_sen = [row[:9] + row[10:15] for row in matrix]
        no_lex_sen = [row[:10] + row[11:15] for row in matrix]
        no_domain = [row[:11] + row[12:15] for row in matrix]
        no_freq = [row[:12] + row[13:15] for row in matrix]
        no_cluster = [row[:13] + row[14] for row in matrix]
        no_outlier = [row[:14] for row in matrix]

        


