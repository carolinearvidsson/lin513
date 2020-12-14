import pickle

class FeatureMatrix:
    '''A feature matrix where rows represent tokens and columns represent its features''' 

    def __init__(self, fclasses, ws):
        self.ws = ws
        self.matrix = []
        self.complexities = []
        self.fclsses = fclasses
        self.__get_feature_methods()
        
    def __get_feature_methods(self):
        '''Returns a list of specific methods from the feature classes. 
        Importantly, these are the only methods in the feature classes 
        whose names do not start with underscore. Their purpose is
        to return features to be used in the feature matrix.
        '''
        self.feature_methods = [getattr(clss, method) for clss in self.fclsses for\
            method in dir(clss) if callable(getattr(clss, method)) and not\
                 method.startswith('_')]
        print('metoder: ', self.feature_methods)

    def populate_matrix(self):
        for e, wordobj in enumerate(self.ws.single_word):
            #print('Getting features for: ', e, wordobj.token)
            features = [feature for method in self.feature_methods for feature in method(wordobj)]
            self.complexities.append(wordobj.complexity)
            self.matrix.append(features)

