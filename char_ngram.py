import pickle
import nltk
from nltk.lm import MLE
from nltk.lm.models import Lidstone
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import padded_everygram_pipeline as pep
from nltk.corpus import brown
from wordspace import WS


class NgramN():
    '''Represents a ngram-probability model. Trains on the brown corpus 
    to calculate uni- and bigram probabilities of given tokens from the 
    CompLex corpus.
    
    Attributes:
        ngram_models = a dict object containing separate uni-, bi- and trigram models trained on the Brown corpus.
        uni: a language model object containing unigram probabilities.
        bi: a language model object containing bigram probabilities.
        tri: a language model object containing trigram probabilities.
    '''
    def __init__(self):

        self.ngram_models = pickle.load(open('data/ngram_models', 'rb'))
        self.uni, self.bi, self.tri = self.ngram_models.values()
    
    def ngram_probs(self, word_object):
        '''Return uni, bi and trigram probabilities for a given token.
        
        Arguments:
            word_object: a Word object'''
        token = word_object.token
        ngram_probabilities = [self.__uni_prob(token), self.__bi_prob(token),
                               self.__tri_prob(token)
                            ]
        return ngram_probabilities

    
    def __uni_prob(self, token):
        '''Calculate and return unigram probability in logspace.
        
        Arguments:
            token: a word string.
        '''
        uni_prob_val = 0
        for char in token:
            uni_prob_val += self.uni.logscore(char)
        return uni_prob_val

    def __bi_prob(self, token):
        '''Calculate and return bigram probability in logspace.
        
        Arguments:
            token: a word string.
        '''
        token = list(pad_both_ends(token, n=2))
        bi_prob_val = 0
        for index, char in enumerate(token):
            if index == 0: #beginning of word
                continue
            else:
                bi_prob_val += self.bi.logscore(char, [token[index-1]])
        return bi_prob_val

    def __tri_prob(self, token):
        '''Calculate and return trigram probability in logspace.

        Arguments:
            token: a word string.
        '''
        token = list(pad_both_ends(token, n=3))
        tri_prob_val = 0
        for index, char in enumerate(token):
            if index < 2 :
                continue
            else:
                tri_prob_val += self.tri.logscore(char, [token[index-2],
                                                         token[index-1]])
        return tri_prob_val
