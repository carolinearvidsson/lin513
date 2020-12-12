import pickle

class FeatureMatrix:
    '''A feature matrix where rows represent tokens and columns represent its features''' 

    def __init__(self, freqdata, fclasses):
        self.matrix = []
        self.complexities = []
        self.fclsses = fclasses
        self.__get_feature_methods()
        
    def __get_feature_methods(self):
        '''Returns a list of specific methods from the feature classes. 
        Importantly, these are the only methods in the feature classes 
        whose names do not start with underscore. Their purpose is
        to return at least one feature to be used in the feature matrix.
        '''
        self.feature_methods = [getattr(clss, method) for clss in self.fclsses for\
            method in dir(clss) if callable(getattr(clss, method)) and not\
                 method.startswith('_')]
        print('metoder: ', self.feature_methods)

    def populate_matrix(self, wordobj):
        features = [feature for method in self.feature_methods for feature in method(wordobj)] #for wordobj in single_word_space] TÄNK PÅ ATT FEATURESARNA 

        self.complexities.append(wordobj.complexity)
        self.matrix.append(features)

