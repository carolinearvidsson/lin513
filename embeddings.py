# Caroline

import pickle
from os import path
import numpy as np
from matplotlib import pyplot
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import sil_score as sil_score
from nltk.stem import WordNetLemmatizer
from bert_embedding import BertEmbedding


class Embeddings:
  '''Generates BERT embeddings and uses them to perform word sense induction
  on observations belonging to the same word type.

  All public methods (these are not prefixed with leading underscore) are used to 
  return one or more feature(s) of a given word object. These features
  are used to predict lexical complexity of a word in context.
  
  Parameters:
  
    ws (worspace object)
      A wordspace class object.
      
    embfile (str)
      The path of a pickle format file holding the embeddings. If file 
      does not exist, it will be created and the process of generating
      embeddings will be initialized.
      For a detailed rescription of the contents of the embedding file,
      see the documentation for local method: __setup.
      
  Attributes:

    self.all_targets (set)
      A set of all target word types in the wordspace object.

    self.wnl (wordnet lemmatizer object)
      Will be used to lemmatize words.
      '''

  def __init__(self, ws, embfile):
    self.all_targets = ws.target_types
    self.embfile = embfile
    self.wnl = WordNetLemmatizer()
    self.__check_existing_file()

  def __check_existing_file(self):
    '''Checks if the pickled file given as class parameter and
    containing the embedding dictionaries already exists. 
    For a detailed description of the dictionaries,
    see documentation of local method: __setup.
    If file does not exist, the retrievement of embeddings is initialized.
    If file already exists, the dicts are loaded into 
    their respective variables and the process of 
    forming clusters for each target word type in
    the data is initialized.
    '''
    if path.exists(self.embfile):
      self.lemma_embs, self.tID_emb = pickle.load(open(self.embfile, "rb"))
      self.__get_average_embedding()
      self.__generate_clusters()
    else:
      self.__setup()
      print('Embeddings have been created and are available \
                      in pickle format at path:', self.embfile)
      print('Embeddings not available:', self.embeddings_na)

  def __setup(self):
    '''Iterates through all sentences in the data in order to 
    get BERT embeddings for every target word instance.
    When all embeddings have been retrieved, they are dumped into
    a pickle format file (path given as class parameter.)
    
    Attributes:

      self.tID_emb (dict)
        Stores target token ID:s (str) as keys and 
        their embeddings (ndarray) as values.
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
        index > 199.
        In the training data of 7000+ target tokens, 
        this happened with a total of 5 tokens.
    '''
    self.tID_emb, self.lemma_embs,  = {}, {}
    self.sentences, self.embeddings_na = set(), []
    self.bert = BertEmbedding(max_seq_length=200)
    for wobj in self.ws.single_word:
      sen, tokn, tID = wobj.sentence, wobj.token.lower(), wobj.id
      print('Getting embeddings for sentence ' + tID + '...')
      self.__populate_embdicts(self.__get_embddngs(sen.split('\n'), tokn, tID))
    pickle.dump([self.lemma_embs, self.tID_emb], open(self.embfile, 'wb'))

  def __populate_embdicts(self, __get_embddngs):
    '''A method that populates the tID_emb dictionary with a sentence's
    particular target token embedding.
    Furthermore, embeddings of all target tokens in the sentence
    (not just the target token of that particular sentence),
    are retrieved as long as the sentence has not previously been parsed,
    and stores them in dictionary lemma_embs.

    For a detailed description of the embedding dictionaries,
    see documentation for local method: __setup.

    Parameters:

      __get_embddngs (funct)
        Provides the following variables:
        tokens (list)
          A tokenized and lowered sentence.
        sen_emb (list)
          Contains the sentence's embeddings (ndarrays). 
        target_emb (ndarray)
          Embedding for that particular sentence's target token.
        tID (str)
          ID number for sentence's target token.
    '''
    tokens, sen_emb, target_emb, tID = __get_embddngs
    self.tID_emb[tID] = target_emb

    if ''.join(tokens) not in self.sentences:
      self.sentences.add(''.join(tokens))
      for wtype in self.all_targets:
        if wtype in tokens:
          # Get index of all target words in the sentence.
          type_indices = [i for i, tokn in enumerate(tokens) if tokn == wtype]
          wtype = self.wnl.lemmatize(wtype)
          for index in type_indices:
            self.lemma_embs.setdefault(wtype, []).append(sen_emb[index])

  def __get_embddngs(self, sentence, token, tID):
    '''Generates BERT embeddings for a given sentence.
    
    Returns:
    
      tokens (list)
        A tokenized sentence. All tokens (str) are lower cased.
      
      sen_emb (list)
        Each element is a token embedding (1D ndarray). These embedding have the same 
        list indices as their correspondent tokens in the 'tokens'-list.
      
      target_emb (ndarray)
        The sentence's particular target token's embedding. 
        If the target token appears more than once in the sentence,
        the embedding of the token with lowest index is retrieved.
        If the target token of that particular sentence cannot be retrieved 
        (this happens if the sentence is longer than 200 tokens and 
        the target has index > 199), the return value is None.

      tID (str)
        The token's ID number.
    '''
    result = self.bert(sentence)
    tokens, sen_emb = result[0][0], result[0][1]
    try:
      target_emb = sen_emb[tokens.index(token)]
    except ValueError:
      self.embeddings_na.append(token + ' is not in sentence ' + tID)
      target_emb = None
    return tokens, sen_emb, target_emb, tID

  def __get_average_embedding(self):
    '''Computes the mean of each column in all the target tokens' 
    embeddings in order to get an "average embedding".
    These dimension means will be returned as features of target tokens
    whose embeddings could not be retrieved during setup.
    '''
    all_target_embeddings = [self.tID_emb[token] for token in \
                self.tID_emb if self.tID_emb[token] is not None]
    self.average_embedding = np.mean(np.array(all_target_embeddings), axis=0)

  def __generate_clusters(self):
    '''Uses the embeddings in self.lemma_embs to generate clusters for 
    each target word type. Stores cluster data (number of clusters and 
    if an embedding is an outlier) to be used 
    as target token features. 
    
    Attributes:

      self.n_clusters (dict)
        Holds lemmatized word types (str) as keys and their estimated
        number of clusters as values. The number of clusters will be 
        greater than 1 only if the 'optimal clustering' 
        generated a silhouette score above 0.25. 
        Number of clusters is defined as the total number of clusters, 
        excluding the number of cluster outliers.
      
      self.cluster_outliers (set)
        Holds all embeddings of tokens that are
        the only members of a cluster.

      pdist_matrx (ndarray)
        A condensed distance matrix. This is needed to create clusters.

      link_matrx (ndarray)
        A hierarchical linkage matrix.
    '''
    self.n_clusters, self.cluster_outliers = {}, set()
    n_clusters = 1
    for wtype in self.lemma_embs:
      if len(self.lemma_embs[wtype]) > 1:
        pdist_matrx = pdist(self.lemma_embs[wtype], metric='cosine')
        link_matrx = linkage(pdist_matrx, method='complete', metric='cosine')

        sil_score, clusters = self.__optimal_clusters(pdist_matrx, link_matrx)
        if sil_score > 0.25:
          n_outliers, outlier_indices = self.__get_outliers(clusters)
          n_clusters = max(clusters) - n_outliers
          for i in outlier_indices:
            self.cluster_outliers.add(tuple(self.lemma_embs[wtype][i]))
      self.n_clusters[wtype] = n_clusters

  def __optimal_clusters(self, pdist_matrx, link_matrx):
    '''Cuts hierarchical clusters at various thresholds and computes
    the silhouette score of the resulting flat clusters in order to find
    the optimal number of clusters. 
    
    Parameters:

      pdist_matrx (ndarray)
        A condensed distance matrix. This is needed to create clusters.

      link_matrx (ndarray)
        A hierarchical linkage matrix.

    Attributes:

      fc (ndarray)
        An array of length N where N is the number of observations.
        C[o] is the cluster number (np.uint64) to which observation o belongs.

      score (float)
        The silhouette score of flattened clusters.
        This is a value between -1 and 1. Values near 1 indicate that
        the flat clusters are well-defined. Values near 0 suggest overlapping
        clusters. Values near -1 indicate that samples within a cluster
        have been assigned to the wrong cluster. 
        Using this program, no silhouette score
        under 0 have been observed. This might be because the flat clusters
        are derived from hierarchical clusters created through
        agglomerative linkage.
        The silhouette score can by definition only be calculated if:
        1 < number of clusters > number of observations.

    Returns:

      max_score (int)
        The highest of all silhouette scores computed
        at the given thresholds.

      best_clusters (ndarray)
        The flat clusters that generated the highest silhouette score.
    '''
    max_score = -1 # Start at lowest possible silhouette score.
    best_clusters = []
    for threshold in np.arange(0.0, 1.0, 0.05):
      fc = fcluster(link_matrx, t=threshold, criterion='distance')
      n_clusters = max(fc)
      if n_clusters != 1 and n_clusters < len(link_matrx):
        score = sil_score(squareform(pdist_matrx), fc, metric='precomputed')
        if score > max_score:
          max_score, best_clusters = score, fc
    return max_score, best_clusters

  def __get_outliers(self, clusters):
    '''Finds if any observation in flat clustering
    is an only member of its cluster. These observations are
    regarded as outliers.

    Parameters:

      clusters (ndarray)
        An array of length N where N is the number of observations.
        C[o] is the cluster number (np.uint64) to which observation o belongs.

      outlier_indices (list)
        Each element is the index (int) of an outlier. 
        Importantly, the indices of each observation correspond to 
        the indices in the list values in self.lemma_embs.

    Returns:
    
      n_outliers (int)
        Number of outliers in the flat clusters.
        
      outlier_indices (list)
        The index (int) of each cluster outlier
    '''
    clusters = clusters.tolist()
    outlier_indices = [clusters.index(obs) for obs in set(clusters) \
                                            if clusters.count(obs) == 1]
    n_outliers = len(outlier_indices)
    return n_outliers, outlier_indices

  def target_emb(self, wobj):
    '''Returns the embedding (list) of a given word object.
    If the embedding could not be retrieved during setup, an
    average embedding is returned
    '''
    embedding = self.tID_emb[wobj.id]
    if embedding is None:
      embedding = self.average_embedding
    return embedding.tolist()

  def n_clusters(self, wobj):
    '''Returns the number of clusters of a given word object's
    lemma (int). If the number of clusters cannot be retrieved
    due to the absence of the word object's embedding, an average
    number of clusters is returned'''
    try:
      lemma = self.wnl.lemmatize(wobj.token.lower())
      return [int(self.n_clusters[lemma])]
    except KeyError:
      all_n_clusters = [self.n_clusters[wtype] for wtype in self.n_clusters]
      average_n_clusters = sum(all_n_clusters) / len(all_n_clusters)
      return [float(average_n_clusters)]

  def is_cluster_outlier(self, wobj):
    '''Returns 1 if a wordobject is a cluster outlier and 0 if it is not.'''
    is_outlier = 0
    embedding = self.tID_emb[wobj.id]
    if embedding is not None and tuple(embedding) in self.cluster_outliers: 
      is_outlier = 1
    return [is_outlier]

# if __name__ == "__main__":
#   from wordspace import WS
#   ws = WS('data/homemade_train.tsv')
#   em = Embeddings(ws, '/Users/carolinearvidsson/homemade_embeddings_train_201214')
    