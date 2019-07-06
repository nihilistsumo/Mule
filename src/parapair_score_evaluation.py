import numpy as np
import tensorflow as tf
import json, os, argparse, copy
from scipy import stats
from collections import Counter

def calculate_accuracy(parapair_scores, parapair_dict, threshold):
    parapair_score_dict = copy.deepcopy(parapair_scores)
    true_parapair_dict = dict()
    max_score = max(list(parapair_score_dict.values()))
    min_score = min(list(parapair_score_dict.values()))
    print("Max in parapair scores: {}".format(max_score))
    print("Min in parapair scores: {}".format(min_score))
    if min_score < 0:
        for pp in parapair_score_dict.keys():
            parapair_score_dict[pp] += abs(min_score)
        max_score = max(list(parapair_score_dict.values()))
    for pp in parapair_score_dict.keys():
        parapair_score_dict[pp] /= max_score
    for page in parapair_dict.keys():
        pairs = parapair_dict[page]['parapairs']
        labels = parapair_dict[page]['labels']
        for i in range(len(labels)):
            true_parapair_dict[pairs[i]] = labels[i]
    for pp in parapair_score_dict.keys():
        parapair_score_dict[pp] = 1 if parapair_score_dict[pp] > threshold else 0
    c = Counter(list(true_parapair_dict.values()))
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    num_pos = c[1]
    num_neg = c[0]
    for pp in true_parapair_dict.keys():
        if pp not in parapair_score_dict.keys():
            p1 = pp.split("_")[0]
            p2 = pp.split("_")[1]
            pp = p2+"_"+p1
        if true_parapair_dict[pp] > 0.99:
            if parapair_score_dict[pp] > 0.99:
                true_pos += 1
            else:
                false_neg += 1
        else:
            if parapair_score_dict[pp] > 0.99:
                false_pos += 1
            else:
                true_neg += 1
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + np.finfo(float).eps)
    precision = true_pos / (true_pos + false_pos + np.finfo(float).eps)
    recall = true_pos / (true_pos + false_neg + np.finfo(float).eps)
    f1 = 2 * precision * recall / (precision + recall + np.finfo(float).eps)

    # print("Positive samples: {}".format(num_pos)+", Negative samples: {}".format(num_neg))
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 measure: {}".format(f1))

def main():
    parser = argparse.ArgumentParser(description="Calculate basic accuracy measures of a parapair score file")
    parser.add_argument("-pp", "--parapair_file", required=True, help="Path to parapair file")
    parser.add_argument("-pps", "--parapair_score", required=True, help="Path to parapair score file")
    parser.add_argument("-t", "--num_threshold", required=True, type=int, help="No. of hreshold values between 0 to 1 for binary classification")
    args = vars(parser.parse_args())
    parapair_file = args["parapair_file"]
    parapair_score_file = args["parapair_score"]
    num_t = args["num_threshold"]
    with open(parapair_file, 'r') as pp:
        parapair = json.load(pp)
    with open(parapair_score_file, 'r') as pps:
        parapair_score = json.load(pps)
    for t in np.linspace(0, 1, num_t):
        print("\nThreshold: {}".format(t))
        calculate_accuracy(parapair_score, parapair, t)

if __name__ == '__main__':
    main()