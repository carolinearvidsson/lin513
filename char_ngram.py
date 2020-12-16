import pickle
import nltk

class Ngram():
    '''Is a character ngram-probability calculator. Utilizes 
    ngram-models previously trained on the Brown corpus with Lidstone 
    smoothing (see model builder in data/ngram_train.py). Uses logspace
    to represent the probability values rather than ...
    
    Attributes:

        ngram_models (dict)
            contains separate uni-, bi- and trigram models trained 
            on the Brown corpus.
        
        uni (dict)
            a language model object containing unigram probabilities.

        bi (dict)
            a language model object containing bigram probabilities.

        tri (dict)
            a language model object containing trigram probabilities.
        
        observed_tokens (dict)
            contains all previously observed tokens as key and their
            probabilities as values, as to avoid repetition.
    '''
    def __init__(self):
        '''Upon initialization, loads previously trained ngram_models.'''

        self.ngram_models = pickle.load(open('data/ngram_models', 'rb'))
        self.uni, self.bi, self.tri = self.ngram_models.values()
        self.observed_tokens = {}
    
    def ngram_probs(self, word_object):
        '''Return list containing uni, bi and trigram probabilities
        in logspace for a given token. For each new token, add token
        and its ngram probabilities to observed_tokens dict.
        
        Parameters:

            word_object (Word-object)
                Represents a single entry in the CompLex corpus.
            
        '''
        token = word_object.token
        if token not in self.observed_tokens:
            ngram_probabilities = [self.__uni_prob(token), self.__bi_prob(token),
                                self.__tri_prob(token)
                                ]
            self.observed_tokens[token] = ngram_probabilities
        else: 
            ngram_probabilities = self.observed_tokens[token]
        return ngram_probabilities

    
    def __uni_prob(self, token):
        '''Calculate and return unigram probability in logspace.
        
        Parameters:

            token (string)
                a single word.
        '''
        uni_prob_val = 0
        for char in token:
            uni_prob_val += self.uni.logscore(char)
        return uni_prob_val

    def __bi_prob(self, token):
        '''Calculate and return bigram probability in logspace.
        
        Parameters:

            token (string)
                a single word.
        '''
        token = list(pad_both_ends(token, n=2)) # tokenize and add (1) start and end symbols
        bi_prob_val = 0
        for index, char in enumerate(token):
            if index == 0: # if beginning of word, skip as to avoid index error
                continue
            else:
                bi_prob_val += self.bi.logscore(char, [token[index-1]]) # get probability of present character, given previous character
        return bi_prob_val

    def __tri_prob(self, token):
        '''Calculate and return trigram probability in logspace.

        Parameters:

            token (string)
                a single word.
        '''
        token = list(pad_both_ends(token, n=3)) # add (2) start and end symbols
        tri_prob_val = 0
        for index, char in enumerate(token):
            if index < 2 : # if first or second character of word, skip as to avoid index error
                continue
            else:
                tri_prob_val += self.tri.logscore(char, [token[index-2], # get probability of present character, given previous two characters
                                                         token[index-1]]) 
        return tri_prob_val
