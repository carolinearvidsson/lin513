import glob
import re
import math

class Frequency:
    '''Represents a frequency lexicon. Google ngram files that constitute the lexicon are
    tab separated text files where each line represents a word type and columns have the following structure:
    1. word type
    2. absolute frequency
    3. number of pages on which the word type occurs
    4. number of books in which the word type occurs 
    '''

    def __init__(self, files):
        self.filenames = glob.glob(files)
        self.frequencies = {}
        self.not_in_freq_data = []

    def get_absfrequency(self, wordobj):
        word = wordobj.token.lower()
        if word not in self.frequencies:
            return self.__parse_external_corpus(word)
        else:
            return [math.log(self.frequencies[word])]

    def __parse_external_corpus(self, word):
        '''Takes a word (str) as input and returns 
        the logarithm of its absolute frequency (int)
        if the word exists in the frequency data.
        '''
        for filename in self.filenames:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                regex = re.compile('^' + word + '\t.+$')
                for line in lines:
                    if regex.match(line) is not None:
                        freqdata = regex.match(line).group().split('\t')
                        absolute_freq = int(freqdata[1])
                        self.frequencies[freqdata[0]] = absolute_freq
                        return [math.log(absolute_freq)]
        
        self.not_in_freq_data.append(word)
        print('frequency not available: ', self.not_in_freq_data)
        return ['n/a'] #PROBLEM frekvens finns inte i datan