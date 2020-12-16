# Christoffer

from word import Word
import numpy as np

class WS:
    '''Represents a collection of Word-objects, which are entries from 
    a subset of the whole CompLex corpus.
    
    Parameters:

        file (str)
            represents a filename, a specific subset of the CompLex corpus.
            File will be tab separated (.tsv).

    Attributes:

        single_word (set) 
            Collection of Word-objects where the target is a single word.
        
        target_types (set)
            Collection of all unique target types occurring in the data.
        
    '''
    def __init__(self, file):

        self.single_word = set()
        self.target_types = set()
        self.structure_data(file)
    
    def structure_data(self, file):
        '''Structure and extract data from CompLex files. Method populates
        WS attributes with Word-objects (single_word) or just target of 
        entry (target_types). 
        
        Parameters: 
            represents a filename, a specific subset of the CompLex corpus.
            File will be tab separated (.tsv).
        '''
        
        text = np.genfromtxt(open(file), delimiter='\t', skip_header=1, 
                                dtype=None, encoding=None, invalid_raise=False, 
                                deletechars="~!@#$%^&*()-=+~\|]}[{';: /?.>,<.", 
                                comments='##')
        for row in text: 
            self.single_word.add(Word(row))
            self.target_types.add(row[3].lower())



