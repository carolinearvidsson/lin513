import pickle
import nltk
from nltk.lm import MLE
from nltk.lm.models import Lidstone
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import padded_everygram_pipeline as pep
from nltk.corpus import brown


class Ngram():
    '''Is a character ngram-probability calculator. Utilizes 
    ngram-models previously trained on the Brown corpus (see, xxx.py).
    
    Attributes:
        ngram_models: a dict object containing separate uni-, bi- and trigram 
                      models trained on the Brown corpus.
        uni: a language model object containing unigram probabilities.
        bi: a language model object containing bigram probabilities.
        tri: a language model object containing trigram probabilities.
    '''
    def __init__(self):

        self.ngram_models = pickle.load(open('data/ngram_models', 'rb'))
        self.uni, self.bi, self.tri = self.ngram_models.values()
    
    def ngram_probs(self, word_object):
        '''Return list containing uni, bi and trigram probabilities, 
        in logspace, for a given token.
        
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
        token = list(pad_both_ends(token, n=2)) # add (1) start and end symbols
        bi_prob_val = 0
        for index, char in enumerate(token):
            if index == 0: # if beginning of word, skip as to avoid index error
                continue
            else:
                bi_prob_val += self.bi.logscore(char, [token[index-1]]) # get probability of present character, given previous character, add to total
        return bi_prob_val

    def __tri_prob(self, token):
        '''Calculate and return trigram probability in logspace.

        Arguments:
            token: a word string.
        '''
        token = list(pad_both_ends(token, n=3)) # add (2) start and end symbols
        tri_prob_val = 0
        for index, char in enumerate(token):
            if index < 2 : # if first or second character of word, skip as to avoid index error
                continue
            else:
                tri_prob_val += self.tri.logscore(char, [token[index-2], # get probability of present character, given previous two characters, add to total
                                                         token[index-1]]) 
        return tri_prob_val
