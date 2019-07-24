import numpy as np
import tensorflow as tf
import json, os, argparse, copy
from scipy import stats
from collections import Counter
from sklearn.metrics import roc_auc_score

def normalize_parapair_scores(parapair_scores):
    parapair_score_dict = copy.deepcopy(parapair_scores)
    max_score = max(list(parapair_score_dict.values()))
    min_score = min(list(parapair_score_dict.values()))
    # print("Max in parapair scores: {}".format(max_score))
    # print("Min in parapair scores: {}".format(min_score))
    if min_score < 0:
        for pp in parapair_score_dict.keys():
            parapair_score_dict[pp] += abs(min_score)
        max_score = max(list(parapair_score_dict.values()))
    for pp in parapair_score_dict.keys():
        parapair_score_dict[pp] /= max_score
    return parapair_score_dict

def calc_auc(parapair_scores, parapair_dict):
    normalized_parapair_scores = normalize_parapair_scores(parapair_scores)
    true_parapair_dict = dict()
    for page in parapair_dict.keys():
        pairs = parapair_dict[page]['parapairs']
        labels = parapair_dict[page]['labels']
        for i in range(len(labels)):
            true_parapair_dict[pairs[i]] = labels[i]
    ytrue = []
    yhat = []
    for pp in normalized_parapair_scores.keys():
        yhat.append(normalized_parapair_scores[pp])
        ytrue.append(true_parapair_dict[pp])
    return roc_auc_score(ytrue, yhat)

def main():
    parser = argparse.ArgumentParser(description="Calculate basic accuracy measures of a parapair score file")
    parser.add_argument("-pp", "--parapair_file", required=True, help="Path to parapair file")
    parser.add_argument("-pps", "--parapair_score", required=True, help="Path to parapair score file")
    # parser.add_argument("-o", "--out", help="Path to measurement output file")
    args = vars(parser.parse_args())
    parapair_file = args["parapair_file"]
    parapair_score_file = args["parapair_score"]
    # outfile = args["out"]
    with open(parapair_file, 'r') as pp:
        parapair = json.load(pp)
    with open(parapair_score_file, 'r') as pps:
        parapair_score = json.load(pps)
    print("AUC score: "+str(calc_auc(parapair_score, parapair)))

if __name__ == '__main__':
    main()