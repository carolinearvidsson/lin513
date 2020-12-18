# Caroline
import glob
import math

class Frequency:
    '''Represents a frequency lexicon.
    
    Parameters:
    
        ws (WS-object) 
            A collection of Word-objects, representing entries 
            in the CompLex corpus.

        files (str)
            Path pattern to Google ngram files (available on the mumin server). 
            Path to files on mumin: 
            /home/corpora/books-ngrams/english/postwar/googlebooks-eng-all-1gram-20090715-*.txt
            Files consist of tab separated values 
            where each line represents a word type
            and columns have the following structure:
                1. word type
                2. absolute frequency
                3. number of pages on which the word type occurs
                4. number of books in which the word type occurs 

    Attributes:
        self.filenames (path)
            All pathnames matching the specified pattern given 
            in the files parameter.

        self.frequencies (dict)
            Keys are word types (str) and values are their frequencies (int).

        self.target_types (set)
            A set of all target words in the wordspace object.
    '''

    def __init__(self, ws, files):
        self.filenames = glob.glob(files)
        self.frequencies = {}
        self.target_types = ws.target_types
        self.__parse_external_corpus()

    def get_absfrequency(self, wordobj):
        '''Takes a word object as argument and
        returns the logarithm of its absolute frequency.
        
        Parameters:
        
            wordobject (Word-object)
                Represents a single entry in the CompLex corpus.
        '''
        word = wordobj.token.lower()
        pseudocount = 0.5
        abs_freq = self.frequencies[word]
        return [math.log(pseudocount + abs_freq)]

    def __parse_external_corpus(self):
        '''Parses through the external frequency corpus and
        retrieves a word's absolute frequency if it exists
        in the wordspace.
        '''
        for filename in self.filenames:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split('\t')
                    if line[0] in self.target_types:
                        self.frequencies[line[0]] = int(line[1])
        self.__not_in_external_corpus()

    def __not_in_external_corpus(self):
        '''Sets the frequency of words missing 
        in the external frequency corpus to 0
        '''
        missing_freq = self.target_types ^ set(self.frequencies)
        for word in missing_freq:
            self.frequencies[word] = 0