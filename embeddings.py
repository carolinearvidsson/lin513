#Requires installment of bert-embedding: pip install bert-embedding
#Saves embeddings in a file on computer
#One sentence at a time takes 7m59.681s. Check time for list of sentences
#Make sure the setup method works before getting train embeddings
#Hur ska jag prioritera? Antingen kommer alla parwise distance matrices finnas lagrade i minnet, eller så tar man fram dem när 
#Det är dags att skapa matriserna varje gång man ska ge ett särdrag till ett token.
#Lemmatize the target words!
#436 of 1252 pairwise dist matrices are empty because 436 of lemmas have only 1 embedding.
#använder nu complete linking. Vill använda ward men då behövs elucidean

import pickle
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, silhouette_samples
from matplotlib import pyplot
from nltk.stem import WordNetLemmatizer
from os import path
from bert_embedding import BertEmbedding
from wordspace import WS

class Embeddings:


  def __init__(self, ws):
    self.ws = ws
    self.all_target_types = set([word_object.token for word_object in self.ws.single_word])
    self.embfile = '/Users/carolinearvidsson/homemade_embeddings_train_201212'
    self.target_not_retrieved = []
    self.pdist_matrices = {}
    self.average_embedding = []
    self.__check_existing_file()

  def get_token_embedding(self, wobj):
    embedding = self.tID_emb[wobj.id]
    if embedding == 'n/a':
      embedding = self.average_embedding
    return embedding.tolist()

  def __get_best_clustering(self, pdist_matrix, linkage_matrix):
    max_score = -1 # Start at lowest possible silhouette score.
    for threshold in np.arange(0.0, 1.0, 0.01):
      flat = fcluster(linkage_matrix, t=threshold, criterion='distance')
      n_clusters = max(flat)
      if n_clusters != 1 and n_clusters < len(linkage_matrix):
        score = silhouette_score(squareform(pdist_matrix), flat, metric='precomputed')
        if score > max_score:
          max_score, best_clusters = score, flat
    try:
      print('max silhouette all: ', max_score)
      print('optimal n_clusters: ', (max(best_clusters)))
      print('best clusters: ', best_clusters)
      print('linkage:\n', linkage_matrix)
    except:
      print('1 cluster')

  def __generate_clusters(self):
    #FIXA
    for e, wtype in enumerate(self.types_emb):
      if e < 2:
        embeddings = list(set(self.types_emb[wtype]))
        pdist_matrix = pdist(self.types_emb[wtype], metric='cosine')
        self.pdist_matrices[wtype] = pdist_matrix
        linkage_matrix = linkage(pdist_matrix, method='complete', metric='cosine')
        fig = pyplot.figure(num=wtype, figsize=(13,5))
        dn = dendrogram(linkage_matrix)
        pyplot.show()
        best_clusters = self.__get_best_clustering(pdist_matrix, linkage_matrix)

  def __check_existing_file(self):
    '''Checks if the pickle file containing the embedding dicts
    (path given in __init__) already exists. 
    If file does not exist the retrievement of embeddings is initialized.
    If file already exists, the dicts are loaded into 
    their respective variables.
    '''
    if path.exists(self.embfile):
      self.types_emb, self.tID_emb = pickle.load(open(self.embfile, "rb"))
      print('Embeddings are available in pickle format at path: ' + self.embfile)
      self.__get_average_embedding()
      #self.__generate_clusters()
    else:
      self.__setup()
      print('Embeddings have been created and are available \
                      in pickle format at path:', self.embfile)
      print('Embeddings not available:', self.target_not_retrieved)

  def __get_average_embedding(self):
    all_target_embeddings = [self.tID_emb[token] for token in\
      self.tID_emb if self.tID_emb[token] != 'n/a']
    self.average_embedding = np.mean(np.array(all_target_embeddings), axis=0)

  def __setup(self):
    '''Iterates through all sentences in the data in order to 
    get BERT embeddings for every target word instance.
    When all embeddings have been retrieved, 
    they are dumped into the embeddings file (path given in __init__).
    
    Attributes:
    
      self.tID_emb (dict)
        Stores target token ID:s (str) as keys and 
        their embeddings (nparray) as values.
    
      self.types_emb (dict)
        Stores lemmatized word types of target tokens (str) as keys. 
        Each value is a list containing embeddings for 
        all occurences of that lemma's base form and inflections.
      
      self.sentences (set)
        Stores sentences (str) that have been parsed.
        Is used to make sure that no embedding duplicates
        are stored in the self.types_emb dictionary.
    '''
    self.tID_emb, self.types_emb, self.sentences = {}, {}, set()
    self.bert = BertEmbedding(max_seq_length=200)
    self.wnl = WordNetLemmatizer()
    for wobj in self.ws.single_word:
      sen, tokn, tID = wobj.sentence, wobj.token.lower(), wobj.id
      print('Getting embeddings for sentence ' + tID + '...')
      self.__populate_embdicts(self.__get_embddings(sen.split('\n'), tokn, tID))
    pickle.dump([self.types_emb, self.tID_emb], open(self.embfile, 'wb'))

  def __populate_embdicts(self, __get_embddings):
    ''''Populates the embedding dicts.
    
    Parameters:
      __get_embddings (funct)
        Returns the following variables:
          tokens (list)
            Tokenized and lowered sentence.

        '''
    tokens, sen_emb, token_embedding, tID = __get_embddings
    self.tID_emb[tID] = token_embedding

    # Do not get embeddings for same sentence twice.
    if ''.join(tokens) not in self.sentences:
      self.sentences.add(''.join(tokens))
      for wtype in self.all_target_types:
        if wtype in tokens:
          type_indices = [i for i, token in enumerate(tokens) if token == wtype]
          wtype = self.wnl.lemmatize(wtype)
          for index in type_indices:
            self.types_emb.setdefault(wtype, []).append(sen_emb[index])

  def __get_embddings(self, sentence, token, tID):
    '''Returns tokenized sentence, tokens embeddings and target token embedding.'''
    result = self.bert(sentence)
    tokens, sen_emb = result[0][0], result[0][1]
    try:
      token_embedding = sen_emb[tokens.index(token)]
    except ValueError:
      self.target_not_retrieved.append(token + ' is not in sentence ' + tID)
      token_embedding = 'n/a'
    return tokens, sen_emb, token_embedding, tID

if __name__ == "__main__":
  ws = WS(['data/homemade_train.tsv'])
  Embeddings(ws)