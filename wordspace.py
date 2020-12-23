# Christoffer

from word import Word
import numpy as np
# Murathan: Great!
class WS:
    '''Represents a collection of Word-objects, which are entries from 
    the whole CompLex corpus.
    
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
        '''Structure and extract data from a file from the CompLex corpus. 
        Method populates. WS attributes with Word-objects (single_word) 
        or specific target of entry (target_types). 
        
        Parameters: 
            
            file (str)
                represents a filename, a specific subset of the CompLex corpus.
                File will be tab separated (.tsv).
        '''
        # First line of CompLex files contains column info, thus skip_header=1. 
        # Further arguments in np.genfromtext as to avoid certain Errors and 
        # Warnings that may loose certain attributes for Word-objects.
        text = np.genfromtxt(open(file), delimiter='\t', skip_header=1, 
                                dtype=None, encoding=None, invalid_raise=False, 
                                comments='##')
        for row in text: 
            self.single_word.add(Word(row))
            self.target_types.add(row[3].lower()) # Murathan: I assume this list is needed frequently so you create this class attribute to access this list efficiently.
                                                  # Murathan: Otherwise, it can be recovered from single_words easily? (e.g. set([word.token.lower() for word in self.ws.single_word]))



