# Caroline
'''
A script for training and testing a model to predict lexical complexity
of single words in context.

Usage: 
python3 main.py mode(test/train) modelfile data(test/train) embeddingfile frequencyfile

mode:
If mode is train, the modelfile will be created or overwritten if it exists.
If mode is test, the modelfile will be used to predict lexical complexity.

data:
Path to train or test data. This is a tsv file with following structure:
    1. Target token ID 
    2. Domain type (e.g. bible)
    3. Sentence in which the target token occurs
    4. Target token
    5. Annotated complexity (a float value between 0.0 and 1.0)

embeddingfile:
Path to file containing the embeddings.

If path does not exist, the file will be created and the 
process of getting embeddings from data will be initialized.
The data structure of the file is a list holding two dictionaries:

    Dict1:  Stores target token ID:s (str) as keys and 
            their embeddings (ndarray) as values.
            This dictionary is utlilized in the get_embeddings
            method.
    
    Dict2:  Stores lemmatized target tokens (str) as keys. 
            Each value is a list containing all embeddings for 
            any occurence of the lemma's base forms or inflections.
            This dictionary will be used for word sense induction.

frequencyfile:
Google ngram files on the mumin server. 
Path to files on mumin: 
/home/corpora/books-ngrams/english/postwar/googlebooks-eng-all-1gram-20090715-*.txt
Files consist of tab separated values where each line represents a word type
and columns have the following structure:
    1. word type
    2. absolute frequency
    3. number of pages on which the word type occurs
    4. number of books in which the word type occurs 
    
This script will fill a wordspace object with train or test data, 
provide feature classes to a feature matrix to be used in training and testing
a linear regression model.
'''
import pickle
import sys
from wordspace import WS
from features import FeatureMatrix
from domainspecificity import DomainSpecificity
from embeddings import Embeddings
from frequency import Frequency
from char_ngram import Ngram
from pos import PosTagger
from length import Length
from regression import MultiLinear

if __name__ == "__main__":
    mode = sys.argv[1]
    model = sys.argv[2]
    data = sys.argv[3]# 'data/homemade_train.tsv'
    embeddings = sys.argv[4]#'/Users/carolinearvidsson/homemade_embeddings_train_201214'
    freqdata = sys.argv[5]#'/Users/carolinearvidsson/googlebooks-eng-all-1gram-20090715-*.txt'

    ws = WS(data)
    fclsses = (Ngram(), Length(), PosTagger(ws), DomainSpecificity(ws), Frequency(freqdata, ws), Embeddings(ws, embeddings))
    matrix = FeatureMatrix(fclsses, ws)
    matrix.populate_matrix()
    reg = MultiLinear()

    if mode == 'train':
        train_model = reg.train_linear_model(matrix)
        pickle.dump(train_model, open(model, 'wb'))
    elif mode == 'test':
        model = pickle.load(open(model, 'rb'))
        reg.predict(model, matrix)