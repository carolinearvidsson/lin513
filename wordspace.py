from word import Word
import numpy as np
import nltk
#nltk.download('punkt')



class WS:
    '''Class contains dict objects that separates single- 
    and multiple-word (2) tokens from the CompLex corpus.

    Attributes:
        single_word: set listing single-word Word-objects. Attributes are ID-number of entry, 
        subcorpus token appears in, sentence, token and complexity value.
        
    '''
    def __init__(self, file):

        self.single_word = set()
        self.target_types = set()
        self.structure_data(file)
    
    def structure_data(self, file):
        '''Structure and extract data from CompLex files.
        
        Arguments: 
            files: list of names CompLex-files.
        '''

            #PROBLEM(LÖST!!), genfromtext feltolkar vissa rader/kolumner i train-filerna (därav try).
            #argumentet comments tog bort fallen där ensam '#' gjorde resten av raden till en och samma kolumn. len av dict är nu samma som träningsdatan
        text = np.genfromtxt(open(file), delimiter='\t', skip_header=1, 
                                dtype=None, encoding=None, invalid_raise=False, 
                                deletechars="~!@#$%^&*()-=+~\|]}[{';: /?.>,<.", 
                                comments='##')
        for row in text: 
            self.single_word.add(Word(row))
            self.target_types.add(row[3].lower())



