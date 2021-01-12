# Caroline
# Murathan: I loved this! However, it feels like you can only have one public method in the feature classes and that has to be used to generate features.
# Murathan: would not it be better to have a certain prefix for those functions (generate_feat_*) which you can use at line 46 so you can freely have other public methods.
# Murathan: It is a minor point, tho. Good work!

class FeatureMatrix:
    '''A feature matrix where rows represent target tokens and 
    columns represent their features to be used in predicting
    lexical complexity of target tokens.
    
    Parameters:

        ws (WS-object) 
            A collection of Word-objects, representing entries 
            in the CompLex corpus.
        
        fclasses (tuple)
            Contains the feature classes. Their public methods are
            called in order to retrieve features.
    
    Attributes:

        self.matrix (list of lists)
            Each nested list is a vector containing the features
            for a specific target word.

        self.complexities (list)
            Each element (float) is the annoteted complexity
            for a specific target word.

        self.id (list)
            Word object id
      ''' 

    def __init__(self, ws, fclasses):
        self.ws = ws
        self.fclsses = fclasses
        self.matrix = []
        self.complexities = []
        self.ids = []
        self.__get_feature_methods()
        
    def __get_feature_methods(self):
        '''Creates a list of all public methods in the feature classes. 
        Importantly, these are the only methods in the feature classes 
        which are not prefixed with leading underscore.
        
        Attributes:
            self.fmethods (list)
                Each element is a public method in one of the feature classes.
        '''
        self.fmethods = [getattr(clss, m) for clss in self.fclsses for \
                    m in dir(clss) if callable(getattr(clss, m)) and not  m.startswith('_')]

    def populate_matrix(self):
        '''Executes the feature methods one by one on each
        word object in wordspace. The feature methods 
        return features to be used in the feature matrix.
        
        Attributes:
            feats (list)
                A vector containing features for a given word object.
                This vector is appended to the feature matrix.
        '''
        for wobj in self.ws.single_word:
            feats = [feat for mthd in self.fmethods for feat in mthd(wobj)]
            try:
                self.complexities.append(wobj.complexity)
            self.ids.append(wobj.id)
            self.matrix.append(feats)

