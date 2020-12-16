# Christoffer
import nltk
import pandas as pd

class PosTagger:
    '''Is a collection of part of speech (PoS) tagged sentences.
    To be used in regression analysis, P
    
    Parameters:
        ws (WS-object) 
            A collection of Word-objects, representing entries 
            in the CompLex corpus.
    
    Attributes:

        tagged_sentences (dict) 
            Contains lists where elements are i) a PoS tagged sentence, 
            ii) index in sentence for a specific target word, iii) a 
            (simplified) PoS-tag of target word. Key is unique entry ID.
            
        single_word (set) 
            Collection of Word-objects (where the token is a single word).
        
        token_index_counter (dict)
            Contains all occuring target word sentence indices (key) and 
            their frequencies (value).

    '''

    def __init__(self, ws):
        '''Upon initialization, tag all sentences in given data.'''
        self.tagged_sentences = {}
        self.single_word = ws.single_word
        self.tag_counter = {}
        self.token_index_counter= {}
        self.upenn_content_tags = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 
                              'NNP','NNPS', 'RB', 'RBR', 'RBS' 'VB', 
                              'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        self.__tag_text()
    
    def __tag_text(self):
        '''Tag sentences using nltk's PoS-tagger which produces a list 
        of tuples with word and PoS-tag. Sentences come from entries in 
        the CompLex corpus. 
        
        In addition to tagging sentences, each sentence has a specific 
        target word. Per sentence, store in list PoS-tag of target word 
        and indexed position in sentence. Also, count i) number of times 
        target words belong to particular PoS, ii) number of times target 
        words appear at particular index in sentences. Populate dict 
        tagged_sentences with sentence ID as key and value [tagged sentence, 
        target index, target PoS]. In case of error, value is None.

        Since the Penn Treebank tag set specifies, for instance, nouns 
        by number, verbs by tense, such tags are collapsed into their 
        "parent" PoS (i.e. 'NN' for all noun versions).

        Method uses dummy variables to identify the chosen PoS

        '''    
        self.pos_counter = {'NN': 0, 'JJ':0, 'RB':0, 'VB':0, 'OT':0 }
        
        # Create dummy variables for chosen PoS.
        dummy_matrix = pd.get_dummies(list(pos_counter.keys()))
        self.dummy_vars = {}
        for i, part in zip(range(5), list(self.pos_counter.keys())):
            self.dummy_vars[part] = list(dummy_matrix.loc[i])
        
        self.average_index, n = 0, 0
        
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        for entry in self.single_word:
            token = entry.token
            sentence = tokenizer.tokenize(entry.sentence)
            try:
                tagged_sent = nltk.pos_tag(sentence)
                tok_index = sentence.index(token)
                tok_pos = tagged_sent[tok_index][1]
                if tok_pos in self.upenn_content_tags:
                    self.pos_counter[tok_pos[:2]] += 1
                else:
                    tok_pos = 'OT'
                    self.pos_counter[tok_pos] += 1
                if tok_index not in self.token_index_counter:
                    self.token_index_counter[tok_index] = 0
                self.token_index_counter[tok_index] += 1 
                self.tagged_sentences[entry.id] = [tagged_sent, tok_index, tok_pos[:2]]
                self.average_index += tok_index
                n += 1
            except:
                self.tagged_sentences[entry.id] = None
        self.average_index = round(self.average_index / n)
     
    def get_pos_len(self, wordobject):
        '''Fetch PoS-tag and sentence length features from dictionary
        tagged_sentences. If value is None, due to erroneous data while
        tagging, default values will be most common PoS-tag and sentence
        length.  

        Parameters:
            
            wordobject (Word-object)
                Represents a single entry in the CompLex corpus.
        
        Returns:
            list containing pos-features (as 5 dummy variables), 
            number of words preceeding target in sentence, number
            of lexical words preceeding target

        '''

        if self.tagged_sentences[wordobject.id] == None:
            max_pos = max(self.pos_counter, key = self.pos_counter.get)
            return self.dummy_vars[max_pos] + [self.average_index, self.average_index]
        else:
            pos = self.__pos(wordobject.id)
            sen_len = self.__sen_len(wordobject.id)
            return pos + sen_len 
    
    def __pos(self, sen_id):
        '''Return ID of PoS-tag of the token represented by the Word object.'''
        pos = self.tagged_sentences[sen_id][2]
        return self.dummy_vars[pos]
    
    def __sen_len(self, sen_id):
        '''Get length of sentence up to (not including) token. 
        Return two values: all_sen_len includes all words preceeding token,
        lex_sen_len counts only lexical/content words. These are defined as 
        
        Arguments:
            wordobject: 
        '''
        sentence = self.tagged_sentences[sen_id]
        all_sen_len = sentence[1]
        lex_sen_len = 0
        token_ind = sentence[1]
        preceeding = sentence[0][:token_ind]
        for word, tag in preceeding:
            if tag in self.upenn_content_tags:
                lex_sen_len += 1

        return [all_sen_len, lex_sen_len]



        # tokenizer = nltk.RegexpTokenizer(r'\w+')
        # for entry in self.single_word:
        #     token = entry.token
        #     sentence = tokenizer.tokenize(entry.sentence)
        #     try:
        #         tagged_sent = nltk.pos_tag(sentence)
        #         tok_index = sentence.index(token)
        #         tok_pos = tagged_sent[tok_index][1]
        #         if tok_pos not in self.pos_id: 
        #             self.tag_counter[tok_pos] = 0
        #             self.pos_id.setdefault(tok_pos, len(self.pos_id))
        #         if tok_index not in self.token_index_counter:
        #             self.token_index_counter[tok_index] = 0
        #         self.tag_counter[tok_pos] += 1
        #         self.token_index_counter[tok_index] += 1 
        #         tok_pos_id = self.pos_id[tok_pos]
        #         self.tagged_sentences[entry.id] = [tagged_sent, tok_index, tok_pos_id]
        #     except:
        #         max_pos = max(self.tag_counter, key=self.tag_counter.get)
        #         max_pos_id = self.pos_id[max_pos]
        #         max_tok_index = max(self.token_index_counter, key=self.token_index_counter.get) 
        #         self.tagged_sentences[entry.id] = [nltk.pos_tag(sentence), max_tok_index, max_pos_id]



