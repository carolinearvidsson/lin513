# kan vi gör aså att en funktion ger tillbaka ett värde i matrisen?
import pickle
from wordspace import WS
from domainspecificity import DomainSpecificity
from embeddings import Embeddings
from frequency import Frequency
from char_ngram import NgramN
from pos import PosTagger
from sen_len import SenLen

class FeatureMatrix:
    '''A feature matrix where rows represent tokens and columns represent its features''' 

    def __init__(self, freqdata, *fclasses):
        self.matrix = []
        self.complexities = []
        self.__get_feature_methods()
        
    def __get_feature_methods(self):
        '''Returns a list of specific methods from the feature classes. 
        Importantly, these are the only methods in the feature classes 
        whose names do not start with underscore. Their purpose is
        to return at least one feature to be used in the feature matrix.
        '''
        self.feature_methods = [getattr(clss, method) for clss in fclsses for\
            method in dir(clss) if callable(getattr(clss, method)) and not\
                 method.startswith('_')]
        print(self.feature_methods)

    def populate_matrix(self, wordobj):
        features = [feature for method in self.feature_methods for feature in method(wordobj)] #for wordobj in single_word_space] TÄNK PÅ ATT FEATURESARNA 

        self.complexities.append(wordobj.complexity)
        self.matrix.append(features)

if __name__ == "__main__":
    train_data = ['data/homemade_train.tsv']
    ws = WS(train_data)
    freqdata = '/Users/carolinearvidsson/googlebooks-eng-all-1gram-20090715-*.txt'
    fclsses = (NgramN(), SenLen(PosTagger(ws)), DomainSpecificity(ws), Frequency(freqdata), Embeddings(ws))
    m = FeatureMatrix(fclsses)
    #for wordobj in ws.single_word:
    #    m.populate_matrix(wordobj)
    #pickle.dump(m, open('matrix_train', 'wb'))
