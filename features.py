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
        '''Creates a list of specific methods from the feature classes. 
        Importantly, these are the only methods in the feature classes 
        whose names do not start with underscore. The purpose of these methods is
        to return features to be used in the feature matrix.
        '''
        self.fmethods = [getattr(clss, m) for clss in self.fclsses for \
                    m in dir(clss) if callable(getattr(clss, m)) and not \
                                                            m.startswith('_')]
        print('metoder: ', self.fmethods)

    def populate_matrix(self):
        for e, wobj in enumerate(self.ws.single_word):
            #print('Getting features for: ', e, wobj.token)
            feats = [feat for mthd in self.fmethods for feat in mthd(wobj)]
            self.complexities.append(wobj.complexity)
            self.matrix.append(feats)

