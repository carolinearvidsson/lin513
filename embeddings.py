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
    self.embfile = '/Users/carolinearvidsson/Desktop/homemade_embeddings_test_201210'
    self.target_not_retrieved = []
    self.pdist_matrices = {}
    self.average_embedding = []
    self.__check_existing_file()

  def get_token_embedding(self, wobj):
    embedding = self.tID[wobj.id]
    if embedding == 'n/a':
      embedding = self.average_embedding
    return embedding.tolist()

  def __get_average_embedding(self):
    all_target_embeddings = [self.tID[token] for token in\
      self.tID if self.tID[token] != 'n/a']
    self.average_embedding = np.mean(np.array(all_target_embeddings), axis=0)

  def __get_best_clustering(self, pdist_matrix, linkage_matrix):
    max_score = -1
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
      self.types_emb, self.tID = pickle.load(open(self.embfile, "rb"))
      print('Embeddings are available in pickle format at path: ' + self.embfile)
      self.__get_average_embedding()
      #self.__generate_clusters()
    else:
      self.tID = {} # Holds specific target token IDs as keys and their token embeddings as values
      self.types_emb = {} # Holds target word types as keys and all their embeddings (not just in target context) as values
      self.__setup()
      print('Embeddings have been created and are available in pickle format at path: ' + self.embfile)
      print(self.target_not_retrieved)

  def __setup(self):
    self.bert = BertEmbedding(max_seq_length=200)
    self.wnl = WordNetLemmatizer()
    for wobj in self.ws.single_word:
      sentence, token, tokenID = wobj.sentence, wobj.token.lower(), wobj.id
      print('Getting embeddings for sentence ' + tokenID + '...')
      self.__populate_embedding_dicts(tokenID, \
        self.__get_embeddings(sentence.split('\n'), token, tokenID)
                                        )
    pickle.dump([self.types_emb, self.tID], open(self.embfile, 'wb'))

  def __populate_embedding_dicts(self, tokenID, __get_embeddings):
    ''''Populates the embedding dicts.'''
    tokens, sen_emb, token_embedding = __get_embeddings
    self.tID[tokenID] = token_embedding

    for wtype in self.all_target_types:
      if wtype in tokens:
        type_indices = [i for i, token in enumerate(tokens) if token == wtype]
        wtype = self.wnl.lemmatize(wtype)
        for index in type_indices:
            if wtype in self.types_emb and \
              any((sen_emb[index] == x).all() for x in self.types_emb[wtype]): continue
            self.types_emb.setdefault(wtype, []).append(sen_emb[index])
            self.types_emb[wtype] = list(dict.fromkeys(self.types_emb[wtype]))

  def __get_embeddings(self, sentence, token, tID):
    '''Returns tokenized sentence, tokens embeddings and target token embedding.'''
    result = self.bert(sentence)
    tokens, sen_emb = result[0][0], result[0][1]
    try:
      token_embedding = sen_emb[tokens.index(token)]
    except ValueError:
      self.target_not_retrieved.append(token + ' is not in sentence ' + tID)
      token_embedding = 'n/a'
    return tokens, sen_emb, token_embedding

if __name__ == "__main__":
  ws = WS(['data/homemade_test.tsv'])
  Embeddings(ws)