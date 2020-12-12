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
        
        multiple_word: dictionary listing multiple-word Word-objects. Key is 
        unique ID-number of token, value is Word object containing subcorpus 
        token appears in, sentence, token and complexity value.

        corpus_types: sets of unique types are listed as sets per subcorpus.
        
    '''

    def __init__(self, files):

        self.single_word = set()
        self.multiple_word = {}
        self.corpus_types = {}
        self.structure_data(files)
    
    def structure_data(self, files):
        '''Structure and extract data from CompLex files.
        
        Arguments: 
            files: list of names CompLex-files.
        '''
        for file in files:
            # try:
                #PROBLEM(LÖST!!), genfromtext feltolkar vissa rader/kolumner i train-filerna (därav try).
                #argumentet comments tog bort fallen där ensam '#' gjorde resten av raden till en och samma kolumn. len av dict är nu samma som träningsdatan
            text = np.genfromtxt(open(file), delimiter='\t', skip_header=1, 
                                    dtype=None, encoding=None, invalid_raise=False, 
                                    deletechars="~!@#$%^&*()-=+~\|]}[{';: /?.>,<.", 
                                    comments='##')
            for row in text: 
                
                self.add_to_corpus(row)
                
                
                # Distinguish between single and multiple word tokens and 
                # place Word-object in appropriate Lexicon-dictionary.
                # if ' ' in row[3]:
                #     self.multiple_word.add(Word(row))   
                # else:
                self.single_word.add(Word(row))
            # except: continue
    
    def add_to_corpus(self, row):
        '''Fill dictionary instance attribute corpus_types with all 
        occurring types in the sentence data, divided by subcorpus.
        
        Arguments:
            row: string object is a full entry from the corpus. 
        '''
        sentence = set(nltk.word_tokenize(row[2].lower()))
        if row[1] not in self.corpus_types:
            self.corpus_types[row[1]] = sentence
        else:
            self.corpus_types[row[1]].update(sentence)




files = ['lcp_single_trial.tsv', 'lcp_multi_trial.tsv']

