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
    train_data = ['data/homemade_train.tsv']
    ws = WS(train_data)
    freqdata = '/Users/carolinearvidsson/googlebooks-eng-all-1gram-20090715-*.txt'
    pos = PosTagger(ws)
    fclsses = (pos, SenLen(PosTagger(pos)), DomainSpecificity(ws), Frequency(freqdata), Embeddings(ws))
    m = FeatureMatrix(freqdata, fclsses)
    for wordobj in ws.single_word:
        m.populate_matrix(wordobj)
    pickle.dump(m, open('matrix_train', 'wb'))

    # train = ger modellen. ska avslutas med att man tar train-delen av regression. som ett argument till train okej min träningsfeaturefil ska heta matrix train, om den redan finns så går man visare och tränar modellen
        
    # test = ger mean abs error. ska alltid göras parametrar modellen och featurematrix på testdatan