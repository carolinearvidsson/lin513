# Christoffer
from nltk.corpus import cmudict

class Length():
    '''Class is a word length (character) and syllable count calculator. 
    Utilizes the Carnegie Mellon Pronouncing Dictionary (CMPD/cmudict),
    accessed through nltk, to tokenize words on a phoneme level.
    
    Attributes:

        syll_dict (dict)
            the CMP dictionary with words as keys and pronounciation
            in list form as values. Some entries have alternative 
            pronounciations, in every case only the first one is used.

    '''

    def __init__(self):
        self.syll_dict = cmudict.dict()
    
    def length(self, wordobject):
        '''Check length and syllable count of a word.
        
        Parameters:
            wordobject (Word-object)
                Represents a single entry in the CompLex corpus.   

        Returns:
            length_syllcount (list)
                Contains integers representing token character length 
                and token syllable count 
        '''
        token = wordobject.token.lower()
        length_syllcount = [len(token), self.__syll_count(token)]
        return length_syllcount
    
    def __syll_count(self, token):
        '''Count amount of syllables in a word. Given a token, CMPD returns 
        a list of phonemes: 'natural' = ['N', 'AE1', 'CH', 'UR0', 'A0', 'L'], 
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
            # vowels in a row (counted as 1).
            vowels = ['a','e','o','u','i', 'y']
            for i, character in enumerate(token):
                if character in vowels:
                    if i == 0: # to avoid index error looking at previous character, if first character is vowel, count as syllabic.
                        syll_count += 1
                    else:
                        if token[i-1] not in vowels:
                            syll_count += 1

        return syll_count
