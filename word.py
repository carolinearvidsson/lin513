# Christoffer
# Murathan: Great!
class Word:
    '''Represents an entry in the CompLex corpus.
    
    Parameters:

        entry (list)
            a single row in the CompLex corpus. Every element represents
            a single column of row which are used as attributes for class.

    Attributes:

        id (str)
            unique ID of entry.

        subcorpus (str)
            domain entry is from (bible, biomed or europarl).

        sentence (str)
            sentence in which the token appears.

        token (str) 
            a single target word.
        
        complexity (str-object, float-value)
            the perceived complexity of token in context by manual 
            annotators (normalized as 0–1, based on a 1–5 Likert scale).
    '''

    def __init__(self, entry):

        self.id = entry[0]
        self.subcorpus = entry[1]
        self.sentence = entry[2]
        self.token = entry[3]
        self.complexity = entry[4]