import pickle
import sys
from wordspace import WS
from features import FeatureMatrix
from domainspecificity import DomainSpecificity
from embeddings import Embeddings
from frequency import Frequency
from char_ngram import NgramN
from pos import PosTagger
from syllable_count import SyllCount
from regression import MultiLinear

if __name__ == "__main__":
    mode = sys.argv[1]
    model = sys.argv[2]
    data = sys.argv[3]#['data/homemade_train.tsv']
    embeddings = sys.argv[4]#'/Users/carolinearvidsson/homemade_embeddings_train_201212'
    freqdata = sys.argv[5]#'/Users/carolinearvidsson/googlebooks-eng-all-1gram-20090715-*.txt'

    ws = WS(data)
    fclsses = (Ngram(), SyllCount(), PosTagger(ws), DomainSpecificity(ws), Frequency(freqdata), Embeddings(ws, embeddings))
    matrix = FeatureMatrix(fclsses, ws)
    matrix.populate_matrix()

    if mode == 'train':
        pickle.dump(matrix, open(model, 'wb'))
    elif mode == 'test':
        reg = MultiLinear()
        reg.get_value(model, matrix)