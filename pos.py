#använder data som den är (i fråga om case). bör få data så vid initiering
#kombinera pos och sen_len? annars görs pos-taggningen två gånger??
#gruppera in alla verb/nouns etc i samma kategori??
from wordspace import WS
import nltk
nltk.download('averaged_perceptron_tagger')

class PosTagger:
    '''Is a collection of part of speech tagged sentences.
    
    Arguments:
        ws: a WS (WordSpace) object, containing Word objects.
    Attributes:
        tagged_sentences: a dictionary with sentence/token-IDs as key. Value is a list with i) a tagged sentence, ii) index for token of interest, iii) the id number representing part of speech of token.
        single_word: a set of Word-objects (where the token is a single word).
        pos_id: a dictionary with PoS-tag as key and an integer as value.
        tag_counter: a dictionary with PoS-tag as key and a count of number of occurrences in tagged material as value.
        token_index_counter: a dictionary with all påträffade index of token in sentence as key, number of occurrences as value. 
    '''

    def __init__(self, ws):
        '''Upon initialization, tag all sentences in given data.'''
        self.tagged_sentences = {}
        self.single_word = ws.single_word
        self.pos_id = {}
        self.tag_counter = {}
        self.token_index_counter= {}
        self.__tag_text()
    
    def __tag_text(self):
        '''Tag sentences using nltk's PoS-tagger which produces a list of tuples 
        with word and PoS-tag. Assign a unique ID (as int) for specific PoS. 
        Count i) number of times a token is a particular PoS, ii) number of times 
        the specific indice where token appears in sentence. 
        
        Fill tagged_sentences dictionary with key being the specific entry's ID, 
        and a key as follows [tagged sentence, index where token is found, ID for PoS]
        In case of error and PoS or index cannot be found (i.e. errors in the 
        given data), give most common index and PoS.

        '''    
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        for entry in self.single_word:
            token = entry.token
            sentence = tokenizer.tokenize(entry.sentence)
            try:
                tagged_sent = nltk.pos_tag(sentence)
                tok_index = sentence.index(token)
                tok_pos = tagged_sent[tok_index][1]
                if tok_pos not in self.pos_id: 
                    self.tag_counter[tok_pos] = 0
                    self.pos_id.setdefault(tok_pos, len(self.pos_id))
                if tok_index not in self.token_index_counter:
                    self.token_index_counter[tok_index] = 0
                self.tag_counter[tok_pos] += 1
                self.token_index_counter[tok_index] += 1 
                tok_pos_id = self.pos_id[tok_pos]
                self.tagged_sentences[entry.id] = [tagged_sent, tok_index, tok_pos_id]
            except:
                max_pos = max(self.tag_counter, key=self.tag_counter.get)
                max_pos_id = self.pos_id[max_pos]
                max_tok_index = max(self.token_index_counter, key=self.token_index_counter.get) 
                self.tagged_sentences[entry.id] = [nltk.pos_tag(sentence), max_tok_index, max_pos_id]
    
    def get_pos(self, wordobject):
        '''Return ID of PoS-tag of the token represented by the Word object.'''
        pos_id = self.tagged_sentences[wordobject.id][2]
        return [pos_id]
    
    def get_sen_len(self, wordobject):
        #kolla upp adverb och particip taggar
        '''Get length of sentence up to (not including) token. 
        Return two values: all_sen_len includes all words preceeding token,
        lex_sen_len counts only lexical/content words. These are defined as 
        
        Arguments:
            wordobject: 
        '''

        upenn_content_tags = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 
                              'NNP','NNPS', 'RB', 'RBR', 'UH', 'VB', 
                              'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

        sen_id = wordobject.id
        sentence = self.tagged_sentences[wordobject.id]
        all_sen_len = sentence[1]

        try:
            lex_sen_len = 0
            token_ind = sentence[1]
            preceeding = sentence[0][:token_ind]
            for word, tag in preceeding:
                if tag in upenn_content_tags:
                    lex_sen_len += 1
        except:
            lex_sen_len = sentence[1]
        return [all_sen_len, lex_sen_len]


