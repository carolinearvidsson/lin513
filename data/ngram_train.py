import pickle
import nltk
from nltk.lm import MLE
from nltk.lm.models import Lidstone
from nltk.lm.preprocessing import padded_everygram_pipeline as pep
from nltk.corpus import brown
from wordspace import WS


class NgramModels():
    '''Class is a collection of trained ngram-models, based on the Brown 
    corpus. Consists of uni-, bi- and trigram probability models. Models use
    nltk's language model with Lidstone smoothing to calculate probabilities. 
    '''
    def __init__(self):
        ngram_trained_models = {}
        training_data = self.__struct_train_data()
        for i in range(1,4):
            n_model = self.__train_lm(i, training_data)
            ngram_trained_models.setdefault(i, n_model)
        pickle.dump(self.ngram_trained_models, open('ngram_models', 'wb'))
    
    def get_ngram_models(n):
        ngram_trained_models = {}
        training_data = struct_train_data(brown.words())
        for i in range(1,n):
            n_model = train_lm(i, training_data)
            ngram_trained_models.setdefault(i, n_model)
        pickle.dump(ngram_trained_models, open('ngram_models', 'wb'))

    def __train_lm(self, n, training_data):
        '''Train a language model object with tokenized training data, 
        smooth the probability matrix with Lidstone.

        Parameters:

            n (int)
                declares what ngram model to be trained.
            
            training_data (list)
                contains lists of (tokenized) words, formatted 
                to fit nltk's language model module.
        '''
        train, vocab = pep(n, training_data)
        lm = MLE(n)
        lm = Lidstone(0.5, lm)
        lm.fit(train, vocab)
        return lm

    def __struct_train_data(self):
        '''Structure training data from the Brown corpus 
        for nltk's language model module. 
        
        Ex: [['w', 'o', 'r', 'd', '1'], ['w', 'o', 'r', 'd', '2'], ...]
        '''
        final_training_data = []
        for word in brown.words():
            final_training_data.append([char for char in word])
        return final_training_data

if __name__ == "__main__":
    get_ngram_models(n)