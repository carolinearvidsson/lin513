# Caroline
'''
A script for training and testing a model to predict lexical complexity
of single words in context.

Usage: 
python3 main.py mode(test/train) modelfile data(test/train) embeddingfile frequencyfile
No arguments are optional.

mode:
If mode is train, modelfile will be created or overwritten if it exists.
If mode is test, modelfile will be loaded and used to predict lexical complexity.

modelfile:
Path to the model file. This file will, depending on mode, be created or
used to predict lexical complexity.

data:
Path to train or test data. 
This is a tsv file where rows represent data points and
columns have the following structure:
    1. Target token ID 
    2. Domain type (e.g. bible)
    3. Sentence in which the target token occurs
    4. Target token
    5. Annotated complexity (a float value between 0.0 and 1.0)

embeddingfile:
Path to file containing the embeddings.

File containing embeddings for both train and trial data 
is available for download on mumin (RECOMMENDED):
/home/lin205_caar5483/lin513/embeddings_train_and_trial

If path does not exist, this file will be created and the 
process of getting embeddings from the given data will 
be initialized (NOT RECOMMENDED).

The data structure of the file is a list holding two dictionaries:

    Dict1:  Stores target token ID:s (str) as keys and 
            their embeddings (ndarray) as values.
    
    Dict2:  Stores lemmatized target tokens (str) as keys. 
            Each value is a list containing all embeddings for 
            any occurence of the lemma's base forms or inflections.

frequencyfile:
Google ngram files (available for download on the mumin server).
Path to files on mumin: 
/home/corpora/books-ngrams/english/postwar/googlebooks-eng-all-1gram-20090715-*.txt
Files consist of tab separated values where each line represents a word type
and columns have the following structure:
    1. word type
    2. absolute frequency
    3. number of pages on which the word type occurs
    4. number of books in which the word type occurs 
    
This script will fill a wordspace object with train or test data, 
define a series of feature classes to be be used in creating a feature matrix.
Depending on the chosen mode (test or train), 
the script uses the feature matrix to either train a regression model 
or predict lexical complexities with an already trained model.
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
    data = sys.argv[3]
    embeddings = sys.argv[4]
    freqdata = sys.argv[5]

    ws = WS(data)
    fclsses = ( Ngram(), Length(), 
                PosTagger(ws), DomainSpecificity(ws), 
                Frequency(ws, freqdata), Embeddings(ws, embeddings)
                )
    matrix = FeatureMatrix(ws, fclsses)
    matrix.populate_matrix()
    reg = MultiLinear()

    if mode == 'train':
        train_model = reg.train_linear_model(matrix)
        pickle.dump(train_model, open(model, 'wb'))
    elif mode == 'test':
        model = pickle.load(open(model, 'rb'))
        reg.predict(model, matrix)