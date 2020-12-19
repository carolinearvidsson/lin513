# lin513

The aim of this program is to predict lexical complexity of single words in context. It collects a series of features per target word and trains a regression model to predict complexities of words in context.

## Data
The dataset consists of a subset of the CompLex corpus ([Shardlow, Cooper and Zampieri, 2020](https://arxiv.org/pdf/2003.07008.pdf)) and was provided as part of [SemEval 2021 (Task 1)](https://sites.google.com/view/lcpsharedtask2021). The data consists of a collection of sentences from multiple domains and for each sentence there is a chosen target word. The sentences' target words are annotated using a 5-point Likert scale (1 very easy – 5 very difficult), and then normalized to a 0 – 1 scale (0 being the least difficult). 

Training and test files are tab separated (.tsv) in which each row represents 
a target word in context, and columns have the the following column structure:
1. Sentence/token ID
2. Domain (e.g. bible, europarl, biomed)
3. Sentence
4. Target word
5. Complexity

### Supplementary data
The folder 'data' in this repository contains different versions of the data set to be used for training and/or testing. It also contains files (ngram_models, ngram_train.py and domainspecific.pickle) needed for running this program. For more information on these files, see sections about feature classes [Ngram](#ngram) and [Domain Specificity](#ds) or read the in-file documentation for these classes.

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

Arguments         | Description
----------------- | ----------------- 
***mode:***        |Can be either `test` or `train` (see section '2.0 Training the model' and '2.1 Testing the model' for explicit examples).
***modelfilepath:***|Path to the file containing the model. If mode is train, this file will be created or overwritten. If mode is test, the model will be used to predict lexical complexity.
***datafilepath:***|Depending on mode, this will be the path to either the train or test data.
***embeddingfilepath:***|Path to the file containing the embeddings. To get this file, see section '1. Get the embedding file'
***frequencyfilespath***|Path the the files constituting the the Google Books 1gram frequencies. For those with access to the mumin server. These files are available for download at path: /home/corpora/books-ngrams/english/postwar/googlebooks-eng-all-1gram-20090715-*.txt. For those without access to mumin, you can get [the data sets here](http://storage.googleapis.com/books/ngrams/books/datasetsv2.html).

#### Testing and training
Let's say you want to name your model file 'trained_model' and you have a training data file named 'train_data.tsv', a testing data file named 'test_data.tsv', a file containing embeddings for test and training data named 'embeddings_train_test' and google 1gram frequency files at path 'google1grams/*.txt'.

To train the model, enter the following in the command line:

`python3 main.py train trained_model train_data.tsv embeddings_train_test google1grams/*.txt`

To test the model, enter the following:

`python3 main.py test trained_model test_data.tsv embeddings_train_test google1grams/*.txt`


### Output (CFS) <a name='output'></a>

The extracted features and manually annotated complexities of the training data will be used to train regression models, at present using Bayesian ridge regression (through scikitlearn). Before training, the program will create a number of versions of the feature matrix with different combinations of features. For this purpose, the features are grouped into two larger categories, embeddings-based (770 features) and handcrafted (14 features). The embeddings-based features are all based on BERT-embeddings and consist of word vectors (768 dimensions) and two cluster-related features. The combinations of features used to build separate models are as follows:

- All features (784 features)
- Only handcrafted (14 features)
- Only embeddings-based (770 features)
- Only handcrafted and cluster-related features (16 features)
- Handcrafted, cluster-related features and 50 embeddings (66 features)

The incoming feature matrix extracted from the test data goes through the same process of version creation. The number of versions and how they are structured can easily be changed in the script (see regression.py). 

As final output when running the program in test mode, the program prints statistic measures from comparing the predicted complexities and the manually annotated complexities found for each target. The statistic measures used are the same as the task authors have published as expected baseline performance on the task's [website](https://github.com/MMU-TDMLab/CompLex). These are Pearson's R, Spearman's Rho, Mean Absolute Error (MAE), Mean Squared Error (MSE) and R-squared (R2). 

The program will execute these measures for each of the trained models. 

## Classes

Below are all the classes used by the program, the majority of which are classes used to extract features from the data. For more detailed descriptions, please refer to the in-file documentation for each class. 

### Basic data structure

Upon running the program, in either training or test mode, the data will be structured by the classes WS and Word. The class FeatureMatrix collects and organizes extracted features and calls on the class MultiLinear to train (train mode) regression models and predict complexities and test (test mode) said models. 

#### WS (Wordspace) (CFS)

The wordspace contains all entries from the given data per mode. It collects unique Word objects (see below) in a set as well as stores all target types which will be used by some of the feature classes.  

#### Word (CFS)

The Word object represents a single entry (i.e. row) from the dataset. The content of each column (see Data section above) are used as attributes.

#### FeatureMatrix (CA)
A feature matrix where rows represent target tokens and columns represent their features to be used in predicting lexical complexity of single words in sentence context.

#### MultiLinear (CFS)

The class is used both for training regression models as well as testing the models, depending on chosen mode. As described in section [Output](#output), it creates versions of the incoming feature matrix to train and test multiple models. Its final output prints results from statistic measures per created model. 

### Features <a name='feat'></a> 

The following features are calculated for each entry's target word. In total there are 784 feature values spread over nine classes. Some of the features are solely based on the target word itself, while some of them take the surrounding sentence context into consideration. All public methods in the feature classes (i.e. not prefixed with leading underscore) return one or more feature(s) of a given word object.

***ATTENTION:*** it is easy to change the number of features by excluding/adding feature classes called in main.py. If this is done, make sure to check the indices used to create versions of the full feature matrix in regression.py. Otherwise, it will make versions of the matrix that may not correspond to features as intended. 


#### Length (CFS)

The public methods of the Length-class returns two features:

1. Word length.
2. Syllable count. 

*Word length* returns the number of characters in target word. 

*Syllable count* returns the number of syllables in target word. The method uses the [Carnegie Mellon University Pronounciation Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict), CMUdict, accessed throguh nltk. Given a word, the dictionary returns a list of corresponding phonemes where vowels are marked with numbers, indicating potential lexical stress, which is taken to indicate syllabic status. 

#### Ngram (CFS) <a name='ngram'></a>

The public method of the Ngram-class returns three features based on ngram probabilities (on character level):

1. Unigram probability
2. Bigram probability
3. Trigram probability 

The class expects that (3) ngram models are previously trained. These can be found in the "data" folder (pickled file `ngram_models`). A training script, `ngram_train.py`, can be found in the same folder. The models are trained on the Brown corpus, using nltk's language model module with Lidstone smoothing. The training script is easily modified to train more (or fewer) ngram models. ***ATTENTION:*** if the amount of models are changed from the present standard of three, the code in char_ngram.py must be modified accordingly, as it is presently specifically written for three models. Also, see [attention note at beginning of Features section](#feat) about feature matrix versions.

#### Frequency (CA)
Represents a frequency lexicon. Its public method returns the logarithm of a word's frequency.

#### PosTagger (CFS)
Upon initialization, the class tags all sentences in data for part of speech (PoS) using a tagger from nltk with Penn Treebank PoS-tags. When called, PosTagger class's public methods returns three features (one of which consists of five variables):

1. Part of speech 
2. Sentence length (all words)
3. Sentence length (lexical/content words)

*Part of speech* returns the PoS of target word through five variables that together indicate the PoS. For each pre-tagged sentence and target word, the method used for the feature finds the target in sentence and thus the PoS. Not all PoS are included as its own variable and feature; nouns, verbs, adjectives and adverbs are classified separately by themselves while all other PoS are combined as 'other'. To represent these categorical features in a regression model, pandas dummy variable module is used using five binary categories/features. 

*Sentence length (all words)* returns the number of words preceeding the target word.

*Sentence length (lexical/content words)* returns the number of lexical/content words preceeding the target word. The definition of lexical/content PoS is similar to the Part of speech feature's categorization of PoS; the PoS deemed lexical are nouns, verbs, adjectives and adverbs. 

#### Domain specificity (CA) <a name='ds'></a>
Generates a set of words that only exist in one of the given domains/supcorpuses (bible, europarl or biomed) in the SemEval (Task 1) training data. During training, a file (data/domainspecific.pickle) containing the domain specific word forms is loaded.

Its public method returns one feature; if a given word object is domain specific or not.

#### BERT embeddings (CA)
Generates BERT embeddings and uses them to perform word sense induction on observations belonging to the same word type.

Its public methods return the following feature types:
1. Token embeddings (each of the 768 dimensions constitutes a single feature)
2. Word type's number of clusters 
3. If token is a cluster outlier (single member of a cluster)



## Scripts

#### ngram_train.py