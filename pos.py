# Christoffer
import nltk
import pandas as pd

class PosTagger:
    '''Class is a collection of part of speech (PoS) tagged sentences 
    and sentence index and PoS of target words. Utilizes nltk's PoS-tagger. 
    
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
        
        upenn_content_tags (list)
            Penn Treebank PoS-tags. Contains tags for PoS deemed to be 
            lexical, or content words.

    '''

    def __init__(self, ws):
        '''Upon initialization, tag all sentences in given data.'''
        self.tagged_sentences = {}
        self.single_word = ws.single_word
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

        For target words in sentence, only 4 PoS are used (and one for 
        others). These are nouns (NN), adjectives (JJ), verbs (VB), adverbs
        (RB) and other (OT). To be able to utilize this feature for regression
        dummy variables (5) created by pandas are used to identify the PoS. 

        Attributes:

            pos_counter (dict)
                Counts the frequency of target word PoS.
            
            dummy_vars (dict)
                Contains the dummy variables for the chosen PoS.

            average_index (int)
                The average index position of the target word over all
                sentences. Utilized if target cannot be found in sentence.

        '''    
        self.pos_counter = {'NN': 0, 'JJ':0, 'RB':0, 'VB':0, 'OT':0 }
        
        # Create dummy variables for chosen PsoS.
        dummy_matrix = pd.get_dummies(list(self.pos_counter.keys())) # Makes as many dummy variables as chosen PoS-tags to consider
        self.dummy_vars = {}
        for i, part in zip(range(len(self.pos_counter)), 
                           list(self.pos_counter.keys())):
            self.dummy_vars[part] = list(dummy_matrix.loc[i])
        
        total_index, n = 0, 0 # Counters to calculate average index of target, in case of error finding target in sentence
        
        # Tokenize text by word to fit nltk's PoS tagger.
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        for entry in self.single_word:
            token = entry.token
            sentence = tokenizer.tokenize(entry.sentence) # pos_tag returns each word and PoS-tag as tuple ('word', tag)
            try:
                tagged_sent = nltk.pos_tag(sentence)
                tok_index = sentence.index(token)
                tok_pos = tagged_sent[tok_index][1] # Find target word in tagged sentences, second element of tuple is PoS-tag
                if tok_pos in self.upenn_content_tags:
                    self.pos_counter[tok_pos[:2]] += 1 # If PoS is considered part of content class, only use first two letters of tag, i.e. "parent" PoS
                else:
                    tok_pos = 'OT' # If PoS not among content classes, make PoS-tag general "other", OT
                    self.pos_counter[tok_pos] += 1
                if tok_index not in self.token_index_counter: # If index of target word in sentence have not previously been seen, create dictionary entry for index
                    self.token_index_counter[tok_index] = 0
                self.token_index_counter[tok_index] += 1 
                self.tagged_sentences[entry.id] = [tagged_sent, tok_index, 
                                                   tok_pos[:2]]
                total_index += tok_index
                n += 1
            except: # If target word cannot be found in sentence, set all values of entry as None
                self.tagged_sentences[entry.id] = None
        self.average_index = round(total_index / n)
     
    def get_pos_len(self, wordobject):
        '''Fetch PoS-tag and sentence length features from dictionary
        tagged_sentences. If value is None, due to erroneous data while
        tagging, default values will be most common PoS-tag and sentence
        length.  

        Parameters:
            
            wordobject (Word-object)
                Represents a single entry in the CompLex corpus.
        
        Returns:
            pos_len
                Contains PoS-features (as 5 dummy variables), number of 
                words preceeding target in sentence, number of lexical words 
                preceeding target. If Wordobject entry has not been able be 
                properly tagged, list will contain the most frequent 
                PoS-features and the average target index (for both sentence 
                length features).

        '''

        if self.tagged_sentences[wordobject.id] == None:
            max_pos = max(self.pos_counter, key = self.pos_counter.get)
            pos_len = self.dummy_vars[max_pos] + [self.average_index, 
                                                  self.average_index]
            return pos_len
        else:
            pos_len = self.__pos(wordobject.id) + self.__sen_len(wordobject.id)
            return pos_len
    
    def __pos(self, sen_id):
        '''Fetch dummy variables representing of PoS-tag of the 
        token represented by the Word object.

        Parameters:
            sen_id (string)
                Unique ID for a wordobject entry.
        
        Returns:
            list of dummy variables (int).
        '''
        pos_tag = self.tagged_sentences[sen_id][2]
        pos_dummy_var = self.dummy_vars[pos_tag]
        return pos_dummy_var
    
    def __sen_len(self, sen_id):
        '''Get length of sentence up to (not including) token. 
        Return two values: all_sen_len includes all words preceeding token,
        lex_sen_len counts only lexical/content words. These are defined as 
        
        Parameters:
            sen_id (string)
                Unique ID for a wordobject entry.
        
        Returns:
            list with two integers representing all words preceeding 
            target word and all lexical/content words preceeding target.
        '''
        sentence_entry = self.tagged_sentences[sen_id]
        all_sen_len = sentence[1] # all_sen_len is same as target's index in sentence, see two lines down
        lex_sen_len = 0
        preceeding = sentence[0][:all_sen_len] # extract only preceeding tuples (containing word and PoS-tag) in sentence
        for word, tag in preceeding:
            if tag in self.upenn_content_tags:
                lex_sen_len += 1 

        return [all_sen_len, lex_sen_len]



