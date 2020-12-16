# Christoffer
from nltk.corpus import cmudict

class Length():
    '''Is a word length (character) and syllable count calculator. 
    Utilizes the Carnegie Mellon Pronouncing Dictionary (CMPD/cmudict) 
    to tokenize words on a phoneme level.
    
    Attributes:

        syll_dict (dict)
            the CMP dictionary with words as keys and pronounciation
            in list form as values. Some entries have alternative 
            pronounciations, in every case only the first one is used.
        
        observed_tokens (dict)
            contains all previously observed tokens as key and their
            word length and syllable count as values, as to avoid repetition.

    '''

    def __init__(self):
        self.syll_dict = cmudict.dict()
        self.observed_tokens = {}
    
    def length(self, wordobject):
        '''Method 
        
        Parameters:

            wordobject (Word-object)
                Represents a single entry in the CompLex corpus.   

        Returns:

            list with integers for word character length and word 
            syllable count 
        '''

        token = wordobject.token.lower()
        if token not in self.observed_tokens:
            length = len(token)
            syll_count = self.__syll_count(token)
            self.observed_tokens[token] = [length, syll_count]
            return [length, syll_count]
        else:
            return self.observed_tokens[token]
    
    def __syll_count(self, token):
        '''Return amount of syllables in a word. CMPD returns a list 
        of phonemes: 'natural' = ['N', 'AE1', 'CH', 'UR0', 'A0', 'L'], 
        where number indicates stress status. This is taken as indicator 
        of syllabic status, thus 'natural' = three syllables.

        Parameters:

            token (str)
                a single word.
        
        Returns:

            syll_count (int)
                number of syllables in token
        
        '''
        syll_count = 0
        try:
            phon_token = self.syll_dict[token]
            for phoneme in phon_token[0]:
                if any(char.isdigit() for char in phoneme) == True:
                    syll_count += 1
        except:
            # If word does not exist in CMPD, make an approximate calculation 
            # of syllable count. Vowel indicates syllable, except multiple 
            # vowels in a row.
            vowels = ['a','e','o','u','i', 'y']
            for i, character in enumerate(token):
                if character in vowels:
                    if i == 0: # to avoid index error, if first character is vowel, count as syllabic.
                        syll_count += 1
                    else:
                        if token[i-1] not in vowels:
                            syll_count += 1

        return syll_count
