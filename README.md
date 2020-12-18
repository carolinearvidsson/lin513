# lin513

The aim of this program is to predict lexical complexity of single words in context. It collects a series of features per target word and trains a regression model to predict complexities of words in context.

## Data
The dataset consists of a subset of the CompLex corpus ([Shardlow, Cooper and Zampieri, 2020](https://arxiv.org/pdf/2003.07008.pdf)) and was provided as part of SemEval 2021 (Task 1). The data consists of a collection of sentences from multiple domains and for each sentence there is a chosen target word. The sentences' target words are annotated using a 5-point Likert scale (1 very easy – 5 very difficult), and then normalized to a 0 – 1 scale (0 being the least difficult). 

Training and test files are tab separated (.tsv) in which each row represents 
a target word in context, and columns have the the following column structure:
1. Sentence/token ID
2. Domain (e.g. bible, europarl, biomed)
3. Sentence
4. Target word
5. Complexity

## Usage

### Setup

### Required installations

- scikit 
- scipy 
- nltk 
- bert-embedding 
- numpy
- pickle

**Get the embedding file (CA)**

In order to run this program, a file containing embeddings for the target words is needed (for
a detailed description of the structure of this file, see documentation in embeddings.py).
For those with acess to the mumin server, the file is available for download at path: 

`/home/lin205_caar5483/lin513/embeddings_train_and_trial`

For those without access to this file, it can be created by first joining the training and test data.
When joining the test and training files, remember to remove the first row of the file that gets appended (this is the row that does not contain a data point, just column labels).
For example, if you append the test file to the train file, the first row in the test file containing column labels should be removed.

Let's say you have a file named 'train_test.tsv', containing both the test and training data.
To create an embedding file named 'embeddings_train_test', run the following code:

```python
from wordspace import WS
from embeddings import Embeddings
ws = WS('train_test.tsv')
em = Embeddings(ws, 'embeddings_train_trial')
```

The process of retrieving embeddings will take approximately 6 hours.

***ATTENTION:*** If you run main.py without having an embedding file at the given path, the retrievement of embeddings will initialize automatically. In this case, embeddings will only be created for the given data file (either test or train). This works, but is not recommended.

### Running the program (CA)

The execution of this program consists of two steps: training and testing. Both steps are done through main.py in the command line. Main takes five arguments as input.

main.py arguments:
1. `mode` 2. `modelfilepath` 3. `datafilepath` 4. `embeddingfilepath` 5. `frequencyfilespath`

Arguments 2, 4 and 5 are identical in the train and test mode (that is if you have an embedding file containing both test and train data), 1 and 3 are not.

***mode:***

Can be either `test` or `train` (see section '2.0 Training the model' and '2.1 Testing the model' for explicit examples).

***modelfilepath:***

Path to the file containing the model. If mode is train, this file will be created or overwritten. If mode is test, the model will be used to predict lexical complexity.

***datafilepath:***

Depending on mode, this will be the path to either the train or test data.

***embeddingfilepath:***

Path to the file containing the embeddings. To get this file, see section '1. Get the embedding file'

***frequencyfilespath***

Path the the files constituting the the Google Books 1gram frequencies.
For those with access to the mumin server. These files are available for download at path:
/home/corpora/books-ngrams/english/postwar/googlebooks-eng-all-1gram-20090715-*.txt

For those without access to mumin, you can get [the data sets here](http://storage.googleapis.com/books/ngrams/books/datasetsv2.html).

#### Testing and training
Let's say you want to name your model file 'trained_model', you have a training data file named 'train_data.tsv', a testing data file named 'test_data.tsv', a file containing embeddings for test and training data named 'embeddings_train_test' and google 1gram frequency files at path 'google1grams/*.txt'.

To train the model, enter the following in the command line:

`python3 main.py train trained_model train_data.tsv embeddings_train_test google1grams/*.txt`

To test the model, enter the following:

`python3 main.py test trained_model test_data.tsv embeddings_train_test google1grams/*.txt`


### Output (CFS)

The extracted features and complexities of the training data will be used to train regression models, at present using Bayesian ridge regression. Before training, the program will create a number of versions of the feature matrix with different combinations of features. These versions are, as program is written now (see further definition below in Features section):

- All features
- Only handcrafted 
- Only features based on BERT-embeddings and the embeddings themselves
- Only embeddings
- Handcrafted, embeddings-based features and 50 embeddings

The incoming features from the test data goes through the same process of creating versions. The number of versions and how they are structured can easily be changed in the script (see regression.py). 

As final output the program prints statistic measures from comparing the predicted complexities and the manually annotated complexities found for each target. The statistic measures used are the same (by type, not necessarily method) as the task authors have published as expected baseline performance on the task's [website](https://github.com/MMU-TDMLab/CompLex). These are Pearson's R, Spearman's Rho, Mean Absolute Error (MAE), Mean Squared Error (MSE) and R-squared (R2). 

The program will execute these measures for each of the trained models. 

## Classes

### Basic data structure

##### WS (Wordspace) (CFS)

The wordspace contains all entries from the given data. It collects unique Word objects (see below) in a set as well as stores all target types.  

##### Word (CFS)

The Word object represents a single entry (i.e. row) from the data given when running the program. The content of each column (see section Data above) is used as an attribute. Word objects are later given to each feature method to extract features.

### Features

The following features are calculated for each entry. In total there are 783 feature values spread over eight overarching feature types. 

#### Handcrafted

##### Word length (CFS)

Returns the number of characters in target word. 

##### Syllable count (CFS)

Returns the number of syllables in target word. Uses the [Carnegie Mellon University Pronounciation Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict), CMUdict, accessed throguh nltk. Given a word, the dictionary returns a list of corresponding phonemes where vowels are marked with numbers, indicating potential lexical stress, which is taken to indicate syllabic status. 

##### Ngram (CFS)

Returns (as code is presently written) three features; uni-, bi- and trigram probabilities on character level for target word. Pre-trained models can be found in the "data" folder (pickled file "ngram_models"), and the training script "ngram_train.py" can easily be modified to train fewer or more models. The training is done using nltk's language model with Lidstone smoothing.  

##### Word frequency (CA)

##### Part of speech (CFS)
Returns five features that together indicate the part of speech of target word. The class utilizes nltk's part of speech tagger (which uses a tagset from Penn Treebank) to tag all sentences in data. 

Utilizes nltk's part of speech tagger (which uses a tagset from Penn Treebank). Represented by dummy variables for parts of speech noun, verb, adjective and adverbs – all other parts of speech are grouped together as other. Returns five features.

##### Domain specificity (CA)

##### Sentence length (CFS)
Consists of two features: number of words (any) preceeding target word and number of lexical/content words preceeding target. Lexical words are here defined as nouns (including proper names), verbs, adjectives and adverbs.

#### Embeddings and word sense induction 

##### BERT embeddings (CA)

##### Clusters and outliers (CA)

## Scripts

#### ngram_train.py