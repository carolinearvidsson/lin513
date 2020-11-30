#tar ca en minut
#bör alla text från brown göras till lower (samt alla tokens till lower)?
#brown lower + tokens lower ≈ brown raw + tokens raw < brown lower + tokens raw, just nu allt raw
#classen kanske tar token list som argument??
# lägg till trigram
# gör till tre olika objekt, ett per gram
# ändra allt till lower
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
        token = word_object.token
        ngram_probabilities = [self.__uni_prob(token), self.__bi_prob(token),
                               self.__tri_prob(token)
                            ]
        return ngram_probabilities

    
    def __uni_prob(self, token):
        '''Calculate and return unigram probability of token.
        
        Arguments:
            token: a word string.
        '''
        uni_prob_val = 0
        for char in token:
            uni_prob_val += self.uni.logscore(char)
        return uni_prob_val

    def __bi_prob(self, token):
        '''Calculate and return list of bigram probabilities in logspace.
        
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
        token = list(pad_both_ends(token, n=3))
        tri_prob_val = 0
        for index, char in enumerate(token):
            if index < 2 :
                continue
            else:
                tri_prob_val += self.tri.logscore(char, [token[index-2],token[index]])
        return tri_prob_val
    
    def __get_tokens(self):
        '''Return all token (strings) from Word objects in the 
        single-word dictionary of a Wordspace object.'''
        # token_list = []
        token_list = [self.single[entry].token for entry in self.single]
        # for entry in self.single:
        #     token_list.append(self.single[entry].token)
        return token_list
