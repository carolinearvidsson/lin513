import pickle
from wordspace import WS
from features import FeatureMatrix
from domainspecificity import DomainSpecificity
from embeddings import Embeddings
from frequency import Frequency
from char_ngram import NgramN
from pos import PosTagger
from sen_len import SenLen

if __name__ == "__main__":
    train_data = ['data/homemade_test.tsv']
    ws = WS(train_data)
    freqdata = '/Users/carolinearvidsson/googlebooks-eng-all-1gram-20090715-*.txt'
    pos = PosTagger(ws)
    fclsses = (pos, SenLen(PosTagger(pos)), DomainSpecificity(ws), Frequency(freqdata), Embeddings(ws))
    m = FeatureMatrix(freqdata, fclsses)
    for wordobj in ws.single_word:
        m.populate_matrix(wordobj)
    pickle.dump(m, open('matrix_test', 'wb'))