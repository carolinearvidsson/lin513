# lin513

The aim of this program is to predict lexical complexity of single words in context.

## Data
The dataset was provided as part of the SemEval 2021 (Task 1) and consists of a collection
of sentences from multiple domains. The sentences' target words are annotated using a 5-point Likert scale (1 very easy – 5 very difficult). 

Training and test files are tab separated (.tsv) and follow the following structure:
1. Sentence/token ID
2. Domain (e.g. bible, europarl, biomed)
3. Sentence
4. Target word
5. Complexity

## Usage
### Required installments
`pip install glob`

`pip install re`

`pip install math`

## Classes
