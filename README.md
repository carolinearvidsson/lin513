# lin513

The aim of this program is to predict lexical complexity of single words in context. It collects a series of features per target words and trains a regression model (using Ridge regularization) to predict complexities of words in context.

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

**1.  Create the embedding file**

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
ws = WS('train_test.tsv')
em = Embeddings(ws, 'embeddings_train_trial')
```

All methods are called through main.py. 

**?.Output**

As final output the program prints statistic measures from comparing the predicted complexities and the manually annotated complexities found for each target in the CompLex corpus.

The statistic measures used are the same (by type, not necessarily method) as the task authors have published as expected baseline performance on the task's [website]()

## Classes

### Basic data structure

##### WS (Wordspace)

The wordspace contains all entries from the given data. It collects unique Word objects (see below) in a set as well as stores all target types.  

##### Word

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







### Required installments

- scikit 
- scipy 
- nltk 
- bert-embedding 
- numpy
- pickle



## Scripts

#### ngram_train.py