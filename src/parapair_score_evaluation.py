import numpy as np
import tensorflow as tf
import json, os, argparse, copy, statistics
from scipy import stats
from collections import Counter
from sklearn.metrics import roc_auc_score

def read_true_parapair_dict(parapair_dict):
    true_parapair_dict = dict()
    for page in parapair_dict.keys():
        pairs = parapair_dict[page]['parapairs']
        labels = parapair_dict[page]['labels']
        for i in range(len(labels)):
            true_parapair_dict[pairs[i]] = labels[i]
    return true_parapair_dict

def normalize_parapair_scores(parapair_scores, norm_method='minmax'):
    parapair_score_dict = copy.deepcopy(parapair_scores)
    if norm_method == 'minmax':
        max_score = max(list(parapair_score_dict.values()))
        min_score = min(list(parapair_score_dict.values()))
        for pp in parapair_score_dict.keys():
            parapair_score_dict[pp] = (parapair_score_dict[pp] - min_score) / (max_score - min_score)
    elif norm_method == 'zscore':
        mean_score = statistics.mean(list(parapair_score_dict.values()))
        std_score = statistics.stdev(list(parapair_score_dict.values()))
        for pp in parapair_score_dict.keys():
            parapair_score_dict[pp] = (parapair_score_dict[pp] - mean_score) / std_score
        parapair_score_dict = normalize_parapair_scores(parapair_score_dict, 'minmax')
    return parapair_score_dict

def calculate_auc(true_parapair_dict, parapair_score_dict):
    ytrue = []
    yhat = []
    for pp in true_parapair_dict.keys():
        ytrue.append(true_parapair_dict[pp])
        yhat.append(parapair_score_dict[pp])
    return roc_auc_score(ytrue, yhat)

def calculate_accuracy(parapair_score_dict, true_parapair_dict, threshold, norm):
    parapair_score_dict = copy.deepcopy(parapair_score_dict)
    # true_parapair_dict = dict()
    # max_score = max(list(parapair_score_dict.values()))
    # min_score = min(list(parapair_score_dict.values()))
    # # print("Max in parapair scores: {}".format(max_score))
    # # print("Min in parapair scores: {}".format(min_score))
    # if min_score < 0:
    #     for pp in parapair_score_dict.keys():
    #         parapair_score_dict[pp] += abs(min_score)
    #     max_score = max(list(parapair_score_dict.values()))
    # for pp in parapair_score_dict.keys():
    #     parapair_score_dict[pp] /= max_score
    # parapair_score_dict = normalize_parapair_scores(parapair_scores, norm)
    # true_parapair_dict = dict()
    # for page in parapair_dict.keys():
    #     pairs = parapair_dict[page]['parapairs']
    #     labels = parapair_dict[page]['labels']
    #     for i in range(len(labels)):
    #         true_parapair_dict[pairs[i]] = labels[i]
    # auc = calculate_auc(true_parapair_dict, parapair_score_dict)
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
    true_pos_rate = true_pos / (true_pos + false_neg + np.finfo(float).eps)
    false_pos_rate = false_pos / (false_pos + true_neg + np.finfo(float).eps)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + np.finfo(float).eps)
    precision = true_pos / (true_pos + false_pos + np.finfo(float).eps)
    recall = true_pos / (true_pos + false_neg + np.finfo(float).eps)
    f1 = 2 * precision * recall / (precision + recall + np.finfo(float).eps)

    # print("Positive samples: {}".format(num_pos)+", Negative samples: {}".format(num_neg))

    print("TPR: {}".format(true_pos_rate)+", FPR: {}".format(false_pos_rate))
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 measure: {}".format(f1))

    return [threshold, true_pos_rate, false_pos_rate, accuracy, precision, recall, f1]

def main():
    parser = argparse.ArgumentParser(description="Calculate basic accuracy measures of a parapair score file")
    parser.add_argument("-pp", "--parapair_file", required=True, help="Path to parapair file")
    parser.add_argument("-pps", "--parapair_score", required=True, help="Path to parapair score file")
    parser.add_argument("-t", "--num_threshold", required=True, type=int, help="No. of threshold values between 0 to 1 for binary classification")
    parser.add_argument("-n", "--normalization", help="Type of normalization to be used (minmax / zscore / no)")
    parser.add_argument("-o", "--out", help="Path to measurement output file")
    args = vars(parser.parse_args())
    parapair_file = args["parapair_file"]
    parapair_score_file = args["parapair_score"]
    num_t = args["num_threshold"]
    norm = args["normalization"]
    outfile = args["out"]
    with open(parapair_file, 'r') as pp:
        parapair = json.load(pp)
    with open(parapair_score_file, 'r') as pps:
        parapair_score = json.load(pps)
    parapair_score_dict = normalize_parapair_scores(parapair_score, norm)
    true_parapair_dict = read_true_parapair_dict(parapair)
    measure_scores = []
    for t in np.linspace(0, 1, num_t):
        print("\nThreshold: {}".format(t))
        measure_scores.append(calculate_accuracy(parapair_score_dict, true_parapair_dict, t, norm))
    print("\nAUC: "+str(calculate_auc(true_parapair_dict, parapair_score_dict)))
    # np.save(outfile, np.array(measure_scores))
    print("Each entry in output score is of the form: [threshold, true_pos_rate, false_pos_rate, accuracy, precision, recall, f1]")

if __name__ == '__main__':
    main()