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
from os import path
import numpy as np
from matplotlib import pyplot
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, silhouette_samples
from nltk.stem import WordNetLemmatizer
from bert_embedding import BertEmbedding


class Embeddings:


  def __init__(self, ws, embfile):
    self.wnl = WordNetLemmatizer()
    self.ws = ws
    self.all_target_types = set([wobj.token.lower() for wobj in self.ws.single_word])
    self.embfile = embfile 

    self.pdist_matrices = {}
    self.average_embedding = []
    self.__check_existing_file()

  def get_token_embedding(self, wobj):
    embedding = self.tID_emb[wobj.id]
    if embedding == 'n/a':
      embedding = self.average_embedding
    return embedding.tolist()

  def get_n_clusters(self, wobj):
    try:
      lemma = self.wnl.lemmatize(wobj.token.lower())
      return [self.cluster_data[lemma]]
    except KeyError:
      all_n_clusters = [self.cluster_data[wtype] for wtype in self.cluster_data]
      average_n_clusters = sum(all_n_clusters) / len(all_n_clusters)
      return [average_n_clusters]

  def is_cluster_outlier(self, wobj):
    is_outlier = 0
    embedding = tuple(self.tID_emb[wobj.id])
    if embedding in self.cluster_outliers: 
      is_outlier = 1
    return [is_outlier]

  def __check_existing_file(self):
    '''Checks if the pickle file containing the embedding dicts
    (path given in __main__) already exists. 
    If file does not exist the retrievement of embeddings is initialized.
    If file already exists, the dicts are loaded into 
    their respective variables.
    '''
    if path.exists(self.embfile):
      self.lemma_embs, self.tID_emb = pickle.load(open(self.embfile, "rb"))
      self.cluster_data, self.cluster_outliers = {}, set()
      self.__get_average_embedding()
      self.__generate_clusters()
      
    else:
      self.__setup()
      print('Embeddings have been created and are available \
                      in pickle format at path:', self.embfile)
      print('Embeddings not available:', self.embeddings_na)

  def __get_average_embedding(self):
    all_target_embeddings = [self.tID_emb[token] for token in \
      self.tID_emb if self.tID_emb[token] != 'n/a']
    self.average_embedding = np.mean(np.array(all_target_embeddings), axis=0)

  def __setup(self):
    '''Iterates through all sentences in the data in order to 
    get BERT embeddings for every target word instance.
    When all embeddings have been retrieved, the process
    of forming clusters for each target word type in
    the data is initialized.
    
    Attributes:

      self.tID_emb (dict)
        Stores target token ID:s (str) as keys and 
        their embeddings (nparray) as values.
        This dictionary is utlilized in the get_embeddings
        method.
    
      self.lemma_embs (dict)
        Stores lemmatized target tokens (str) as keys. 
        Each value is a list containing all embeddings for 
        any occurence of the lemma's base forms or inflections.
        This dictionary will be used for word sense induction.
      
      self.sentences (set)
        Stores sentences (str) that have been parsed.
        Is used to make sure that no embedding duplicates
        are stored in the self.lemma_embs dictionary.

      self.embeddings_na (list):
        Holds the ID:s (str) of tokens that could not be retrieved.
        This can happen when the sentence is longer than 200
        tokens (BERT's max sequence length) and the target token 
        is near the end of sentence.
        In the training data of 7000+ target tokens, 
        this happened with a total of 5 tokens.
    '''
    self.tID_emb, self.lemma_embs, self.sentences = {}, {}, set()
    self.embeddings_na = []
    self.bert = BertEmbedding(max_seq_length=200)
    for wobj in self.ws.single_word:
      sen, tokn, tID = wobj.sentence, wobj.token.lower(), wobj.id
      print('Getting embeddings for sentence ' + tID + '...')
      self.__populate_embdicts(self.__get_embddngs(sen.split('\n'), tokn, tID))
    pickle.dump([self.lemma_embs, self.tID_emb], open(self.embfile, 'wb'))

  def __populate_embdicts(self, __get_embddngs):
    '''A method that populates the tID_emb dictionary with a sentence's
    particular target token embedding.
    Furthermore, it gets each embedding of any target token in the sentence,
    as long as the sentence has not previously been parsed,
    and stores it in dictionary lemma_embs.

    For a detailed description of the embedding dictionaries,
    see documentation for local method: __setup.

    Parameters:

      __get_embddngs (funct)
        Provides the following variables:
        tokens (list)
          A tokenized and lowered sentence.
        sen_emb (list)
          Contains the sentence's embeddings (nparrays). 
        token_embedding (nparray)
          Embedding for that particular sentence's target token.
        tID (str)
          ID number for sentence's target token.
    '''
    tokens, sen_emb, token_embedding, tID = __get_embddngs
    self.tID_emb[tID] = token_embedding

    if ''.join(tokens) not in self.sentences:
      self.sentences.add(''.join(tokens))
      for wtype in self.all_target_types:
        if wtype in tokens:
          type_indices = [i for i, token in enumerate(tokens) if token == wtype]
          wtype = self.wnl.lemmatize(wtype)
          for index in type_indices:
            self.lemma_embs.setdefault(wtype, []).append(sen_emb[index])

  def __get_embddngs(self, sentence, token, tID):
    '''Returns tokenized sentence, tokens embeddings and target token embedding.'''
    result = self.bert(sentence)
    tokens, sen_emb = result[0][0], result[0][1]
    try:
      token_embedding = sen_emb[tokens.index(token)]
    except ValueError:
      self.embeddings_na.append(token + ' is not in sentence ' + tID)
      token_embedding = 'n/a'
    return tokens, sen_emb, token_embedding, tID

  def __generate_clusters(self):
    n_clusters = 1
    for wtype in self.lemma_embs:
      if len(self.lemma_embs[wtype]) > 1:
        pdist_matrix = pdist(self.lemma_embs[wtype], metric='cosine')
        self.pdist_matrices[wtype] = pdist_matrix
        linkage_matrix = linkage(pdist_matrix, method='complete', metric='cosine')
      #fig = pyplot.figure(num=wtype, figsize=(13,5))
      #dn = dendrogram(linkage_matrix)
      #pyplot.show()
      #print(wtype)
        sil_score, clusters = self.__get_best_clustering(pdist_matrix, linkage_matrix)
        if sil_score > 0.25:
          n_outliers, outlier_indices = self.__get_outliers(clusters)
          n_clusters = max(clusters) - n_outliers
          for i in outlier_indices:
            self.cluster_outliers.add(tuple(self.lemma_embs[wtype][i]))
      self.cluster_data[wtype] = n_clusters

  def __get_best_clustering(self, pdist_matrix, linkage_matrix):
    max_score = -1 # Start at lowest possible silhouette score.
    best_clusters = []
    for threshold in np.arange(0.0, 1.0, 0.05):
      flat = fcluster(linkage_matrix, t=threshold, criterion='distance')
      n_clusters = max(flat)
      if n_clusters != 1 and n_clusters < len(linkage_matrix):
        score = silhouette_score(squareform(pdist_matrix), flat, metric='precomputed')
        if score > max_score:
          max_score, best_clusters = score, flat
    return max_score, best_clusters

  def __get_outliers(self, clusters):
    clusters = clusters.tolist()
    outlier_indices = [clusters.index(obs) for obs in set(clusters) if clusters.count(obs) == 1]
    n_outliers = len(outlier_indices)
    return n_outliers, outlier_indices

# if __name__ == "__main__":
#   from wordspace import WS
#   ws = WS('data/homemade_train.tsv')
#   em = Embeddings(ws, '/Users/carolinearvidsson/homemade_embeddings_train_201213')
#   for wobj in ws.single_word:
#     print(wobj.token, em.get_n_clusters(wobj), em.is_cluster_outlier(wobj))