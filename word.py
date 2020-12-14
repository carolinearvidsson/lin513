# Christoffer

class Word:
    '''Represents an entry of a target word from the CompLex corpus.
    
    Attributes:
        id: identity number of entry.
        subcorpus: domain sentence is from (bible, biomed or europarl).
        sentence: sentence in which the token appears.
        token: a single target word.
        complexity: the perceived complexity of token in context by manual 
                    annotators (normalized as 0–1, based on a 1–5 Likert scale).
    '''

    def __init__(self, entry):

        self.id = entry[0]
        self.subcorpus = entry[1]
        self.sentence = entry[2]
        self.token = entry[3]
        self.complexity = entry[4]

    def get_data(self):
        '''Return 4-tuple with the object's attributes.'''
        return self.id, self.subcorpus, self.sentence, self.token, self.complexity
       