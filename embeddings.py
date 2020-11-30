#Requires installment of bert-embedding: pip install bert-embedding
#Save embeddings in a file on computer
# How about having two classes inside embeddings class? One that generates embeddings and saves them into a file
# and one that reads the files and gets the embedding features?
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
    self.embeddingfile = '/Users/carolinearvidsson/Desktop/embeddings_train_201124'
    self.target_not_retrieved = []
    self.pdist_matrices = {}
    self.__check_existing_file()

  def get_token_embedding(self, wobj):
    return [self.tokenID_embeddings[wobj.id]]

  def __get_best_clustering(self, pdist_matrix, linkage_matrix):
    max_score = 0
    for threshold in np.arange(0.0, 1.0, 0.01):
      flat = fcluster(linkage_matrix, t=threshold, criterion='distance')
      n_clusters = max(flat)
      if n_clusters != 1 and n_clusters < len(linkage_matrix):
        score = silhouette_score(squareform(pdist_matrix), flat, metric='precomputed')
        sample_scores = silhouette_samples(squareform(pdist_matrix), flat, metric='precomputed')
        print(sample_scores)
        if score > max_score:
          max_score, best_clusters = score, flat
    try:
      print('max silhouette all: ', max_score)
      print('optimal n_clusters: ', (max(best_clusters)))
    except:
      print('1 cluster')

  def __generate_clusters(self):
    for e, wordtype in enumerate(self.types_embeddings):
      if e < 6:
        pdist_matrix = pdist(self.types_embeddings[wordtype], metric='cosine')
        self.pdist_matrices[wordtype] = pdist_matrix
        linkage_matrix = linkage(pdist_matrix, method='complete', metric='cosine')
        fig = pyplot.figure(num=wordtype, figsize=(13,5))
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
    if path.exists(self.embeddingfile):
      self.types_embeddings, self.tokenID_embeddings = pickle.load(open(self.embeddingfile, "rb"))
      print('Embeddings are available in pickle format at path: ' + self.embeddingfile)
      self.__generate_clusters()
    else:
      self.tokenID_embeddings = {} # Holds specific target token IDs as keys and their token embeddings as values
      self.types_embeddings = {} # Holds wordtypes of target tokens as keys and all their embeddings (not just in target context) as values
      self.__setup()
      print('Embeddings have been created and are available in pickle format at path: ' + self.embeddingfile)
      print(self.target_not_retrieved)

  def __setup(self):
    self.bert = BertEmbedding(max_seq_length=200)
    for wobj in self.ws.single_word:
      sentence, token, tokenID = wobj.sentence, wobj.token.lower(), wobj.id
      print('Getting embeddings for sentence ' + tokenID + '...')
      self.__populate_embedding_dicts(tokenID, \
        self.__get_embeddings(sentence.split('\n'), token, tokenID)
                                        )
    pickle.dump([self.types_embeddings, self.tokenID_embeddings], open(self.embeddingfile, 'wb'))

  def __get_embeddings(self, sentence, token, tokenID):
    '''Returns tokenized sentence, tokens embeddings and target token embedding.'''
    result = self.bert(sentence)
    tokens, tokens_embeddings = result[0][0], result[0][1] #result[1][0] skulle visa andra meningen tokeniserad. Fixa så modellen läser in mer än en mening i taget.
    try:
      token_embedding = tokens_embeddings[tokens.index(token)]
    except ValueError:
      self.target_not_retrieved.append(token + ' is not in sentence ' + tokenID)
      token_embedding = 'n/a'
    return tokens, tokens_embeddings, token_embedding

  def __populate_embedding_dicts(self, tokenID, __get_embeddings):
    ''''Populates the embedding dicts.'''
    tokens, tokens_embeddings, token_embedding = __get_embeddings
    self.tokenID_embeddings[tokenID] = token_embedding
    lemmatizer = WordNetLemmatizer()
    for wordtype in self.all_target_types:
      if wordtype in tokens:
        type_indices = [i for i, token in enumerate(tokens) if token == wordtype]
        for index in type_indices:
          self.types_embeddings.setdefault(lemmatizer.lemmatize(wordtype), []).append(tokens_embeddings[index])


if __name__ == "__main__":
  ws = WS(['lcp_single_train.tsv'])
  Embeddings(ws)