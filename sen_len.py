from wordspace import WS
import nltk
#nltk.download('averaged_perceptron_tagger')

#penn bank tagset https://www.sketchengine.eu/penn-treebank-tagset/

#3 rader får ingen matchning mellan token och sentence

class SenLen():
    '''Count sentence length up to (not including) token/target word. 
    In case word appears multiple times, the first occurrence will 
    be indexed. Returns two values per token:
    1. Only lexical/content words will be regarded, i.e. adjectives, 
    nouns and proper names, verbs (in case of particle verbs, both 
    parts will be counted) and adverbs.
    2. All words, both lexical/content and grammatical/function words 
    will be counted. 
    
    Attributes:
        single: a dictionary of single-word Word objects from the CompLex corpus.
        upenn_content_tags: a list of PoS-tags for content words from 
                            the UPenn tagset.
    '''

    def __init__(self, pos_object): #(self, WS-object)

        self.upenn_content_tags = ['CD','JJ', 'JJR', 'JJS', 'NN', 'NNS', 
                                   'NNP','NNPS', 'RB', 'RBR', 'UH', 'VB', 
                                   'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        
        self.tagged_sentences = pos_object.tagged_sentences
        #self.get_sen_len()
    
    def get_sen_len(self, wordobject):
        '''Get length of sentence up to and including the token. Only count 
        lexical PoS/content words (tagset in upenn_content_tags).'''
        sen_id = wordobject.id
        sentence = self.tagged_sentences[sen_id]
        all_sen_len = sentence[1]

        try:
            lex_sen_len = 0
            token_ind = sentence[1]
            preceeding = sentence[0][:token_ind]
            for word, tag in preceeding:
                if tag in self.upenn_content_tags:
                    lex_sen_len += 1
        except:
            lex_sen_len = sentence[1]
        return [all_sen_len, lex_sen_len]