# lin513

The aim of this program is to predict lexical complexity of single words in context. It collects a series of features and trains a regression model (using Ridge regularization) to predict complexities of words in context.

## Data
The dataset consists of a subset of the CompLex corpus ([Shardlow, Cooper and Zampieri, 2020](https://arxiv.org/pdf/2003.07008.pdf)) and was provided as part of the SemEval 2021 (Task 1). The data consists of a collection of sentences from multiple domains and for each sentence there is a chosen target word. The sentences' target words are annotated using a 5-point Likert scale (1 very easy – 5 very difficult), and then normalized to a 0 – 1 scale (0 being the least difficult). 

Training and test files are tab separated (.tsv) and follow the following column structure:
1. Sentence/token ID
2. Domain (e.g. bible, europarl, biomed)
3. Sentence
4. Target word
5. Complexity

## Usage

All methods are called through main.py. 

## Features

The following features are calculated for each entry. In total there are 783 feature values spread over eight overarching features. 

### Handcrafted

#### Word length (CFS)

#### Syllable count (CFS)

#### Ngram (CFS)

Consists of three features; uni-, bi- and trigram probabilities on character level. Ngram models are trained with nltk language model (Lidstone smoothing) and returns values in log2.

#### Word frequency (CFS)

#### Part of speech (CFS)
Utilizes nltk's part of speech tagger (which uses a tagset from Penn Treebank). Represented by dummy variables for parts of speech noun, verb, adjective and adverbs – all other parts of speech are grouped together as other. 

#### Domain specificity (CA)

#### Sentence length (CFS)
Consists of two features: number of words (any) preceeding target word and number of lexical/content words preceeding target. Lexical words are here defined as nouns (including proper names), verbs, adjectives and adverbs.

### Embeddings and word sense induction 

#### BERT embeddings (CA)

#### Clusters and outliers (CA)







### Required installments

- scikit 
- scipy 
- nltk 
- bert-embedding 
- numpy
- pickle



## Scripts

