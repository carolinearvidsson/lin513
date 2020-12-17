# Caroline
import pickle

class FeatureMatrix:
    '''A feature matrix where rows represent target tokens and 
    columns represent their features to be used in predicting
    lexical complexity of single words in sentence context.
    ''' 

    def __init__(self, ws, fclasses):
        self.ws = ws
        self.matrix = []
        self.complexities = []
        self.fclsses = fclasses
        self.__get_feature_methods()
        
    def __get_feature_methods(self):
        '''Creates a list of all public methods in the feature classes. 
        Importantly, these are the only methods in the feature classes 
        which do not contain prefixed leading underscore. The feature methods 
        return features to be used in the feature matrix.
        '''
        self.fmethods = [getattr(clss, m) for clss in self.fclsses for \
                    m in dir(clss) if callable(getattr(clss, m)) and not \
                                                            m.startswith('_')]
        print('metoder: ', self.fmethods)

    def populate_matrix(self):
        
        for wobj in self.ws.single_word:
            feats = [feat for mthd in self.fmethods for feat in mthd(wobj)]
            self.complexities.append(wobj.complexity)
            self.matrix.append(feats)

