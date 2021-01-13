import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import csv

def every_corr(matrix, compl):
    matrix = matrix_by_feature(matrix)
    with open('single_features_pearson.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for n, feature in enumerate(matrix):
            if n == 16:
                writer.writerow(['from n=16 on', 'embeddings'])
            corr, _ = pearsonr(feature, compl)
            writer.writerow([n, corr])

def matrix_by_feature(matrix):
    new_matrix = []
    feat_n = len(matrix[0])
    for i in range(feat_n):
        feature_row = []
        for row in matrix:
            feature_row.append(row[i])
        new_matrix.append(feature_row)
    return new_matrix