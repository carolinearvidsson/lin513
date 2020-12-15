# Caroline Arvidsson
import glob
import re
import math

class Frequency:
    '''Represents a frequency lexicon. Google ngram files that are used to fill the lexicon are
    tab separated text files where each line represents a word type and columns have the following structure:
    1. word type
    2. absolute frequency
    3. number of pages on which the word type occurs
    4. number of books in which the word type occurs 
    '''

    def __init__(self, files, ws):
        self.filenames = glob.glob(files)
        self.frequencies = {}
        self.target_types = ws.target_types
        self.__parse_external_corpus()
        self.__not_in_external_corpus()

    def get_absfrequency(self, wordobj):
        word = wordobj.token.lower()
        pseudocount = 0.5
        abs_freq = self.frequencies[word]
        return [math.log(pseudocount + abs_freq)]

    def __parse_external_corpus(self):
        '''Takes a word (str) as input and returns 
        the logarithm of its absolute frequency (int)
        if the word exists in the frequency data.
        '''
        for filename in self.filenames:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split('\t')
                    if line[0] in self.target_types:
                        self.frequencies[line[0]] = int(line[1])

    def __not_in_external_corpus(self):
        missing_freq = self.target_types ^ set(self.frequencies)
        for word in missing_freq:
            self.frequencies[word] = 0