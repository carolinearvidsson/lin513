import pickle
import nltk
from nltk.lm import MLE
from nltk.lm.models import Lidstone
from nltk.lm.preprocessing import padded_everygram_pipeline as pep
from nltk.corpus import brown


def get_ngram_models(n, name):
    '''Train a collection of character ngram-models using nltk's language 
    model with Lidstone smoothing. Data used is from the Brown corpus, 
    accessed through nltk. Pickles a dictionary that contains the models.

    Parameters:

        n (int)
            Defines the highest ngram-model to make. Function will 
            train n amount of models, from unigram to specified ngram.

        name (string)
            The ngram models, collected in a dictionary, will be pickled
            with string as filename.  
    '''
    ngram_trained_models = {}
    training_data = __struct_train_data()
    for i in range(1, n+1):
        n_model = __train_lm(i, training_data)
        ngram_trained_models.setdefault(i, n_model)
    pickle.dump(ngram_trained_models, open(name, 'wb'))


def __struct_train_data():
    '''Structure training data from the Brown corpus 
    for nltk's language model module.

    Returns:

        final_training_data (list) 
            contains lists of character tokenized words. 
            Ex: [['w', 'o', 'r', 'd', '1'], ['w', 'o', 'r', 'd', '2'], ...]
    '''
    final_training_data = []
    for word in brown.words():
        final_training_data.append([char for char in word])
    return final_training_data

def __train_lm(n, training_data):
    '''Train a language model object with tokenized training data, 
    smooth the probability matrix with Lidstone.

    Parameters:

        n (int)
            declares what ngram model to be trained.
        
        training_data (list)
            contains lists of (tokenized) words, formatted 
            to fit nltk's language model module.
    
    Returns:

        lm (Lidstone object)
            A trained character ngram-model.
    '''
    train, vocab = pep(n, training_data)
    lm = MLE(n)
    lm = Lidstone(0.5, lm)
    lm.fit(train, vocab)
    return lm