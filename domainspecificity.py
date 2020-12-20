# Caroline
import pickle
from os import path
from nltk import word_tokenize

class DomainSpecificity:
    '''Generates a set of words that only exist in 
    one of the given domains/supcorpuses (bible, europarl or biomed)
    in the SemEval (Task 1) training data.
    
    Parameters:
        
        ws (WS-object) 
            A collection of Word-objects, representing entries 
            in the CompLex corpus.

    Attributes:
        
        self.domain_specific (set)
            A set of domain specific words.

        self.dsfilepath (str)
            Path to the file containing self.domain_specific.
            If the file does not exist, it will be created.
            If the file exists, it will be loaded and used in 
            getting features for computing lexical complexity.
    '''

    def __init__(self, ws):
        self.ws = ws
        self.domain_specific = set()
        self.dsfilepath = 'data/domainspecific.pickle'
        self.__if_file_exists()

    def is_domain_specific(self, wordobj):
        '''Takes a word object as argument and returns 1 if its token
        is domain specific and 0 if it's not.'''
        is_specific = 0
        if wordobj.token.lower() in self.domain_specific:
            is_specific = 1
        return [is_specific]

    def __if_file_exists(self):
        '''Checks if the file containing the set of domain specific words 
        exists. If it exists, it will be loaded and used to
        retrieve features in self.is_domain_specific. If it does not exist,
        it will be created.
        '''
        if path.exists(self.dsfilepath):
            self.domain_specific = pickle.load(open(self.dsfilepath, "rb"))
        else:
            self.__get_domain_specific(self.__get_domains())
            pickle.dump(self.domain_specific, open(self.dsfilepath, 'wb'))
            print('Domain specific word types have been dumped and are \
            available at path:', self.dsfilepath)

    def __get_domain_specific(self, domains):
        '''Calculates the symmetric difference of sets in order to
        generate a set with word types that only occur 
        in one of the given domains.
        
        Parameters:

            domains (dict)
                Each key (str) is a domain/subcorpus. Their values (set)
                contain all word types (str) that occur in 
                that particular domain.
        '''
        for domain in domains:
            self.domain_specific = self.domain_specific ^ domains[domain]


    def __get_domains(self):
        '''Parses through the sentences in wordspace and
        gets all the words that occur in those sentences.

        Returns:
            domains (dict)
                Each key (str) is a domain/subcorpus. Their values (sets)
                contain all word types (str) that occur in 
                that particular domain.
        '''
        domains = {}
        for entry in self.ws.single_word:
            domain = entry.subcorpus
            sentence = word_tokenize(entry.sentence)
            for word in sentence:
                domains.setdefault(domain, set()).add(word.lower())
        return domains