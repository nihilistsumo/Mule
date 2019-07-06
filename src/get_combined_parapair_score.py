#!/usr/bin/python3

import numpy as np
import tensorflow as tf
import json, os, argparse, random
from scipy import stats

def rev(parapair):
    p1 = parapair.split("_")[0]
    p2 = parapair.split("_")[1]
    return p2+"_"+p1

def load_parapairs(parapair_file):
    with open(parapair_file, 'r') as ppf:
        parapairs = json.load(ppf)
    return parapairs

def normalize_parapair_scores(parapair_score_dict):
    max_score = max(list(parapair_score_dict.values()))
    min_score = min(list(parapair_score_dict.values()))
    if min_score < 0:
        for pp in parapair_score_dict.keys():
            parapair_score_dict[pp] += abs(min_score)
        max_score = max(list(parapair_score_dict.values()))
    for pp in parapair_score_dict.keys():
        parapair_score_dict[pp] /= max_score
    return parapair_score_dict

def zscore_normalize_parapair_scores(parapair_score_dict):
    mu = np.mean(list(parapair_score_dict.values()))
    sigma = np.std(list(parapair_score_dict.values()))
    for pp in parapair_score_dict.keys():
        parapair_score_dict[pp] = (parapair_score_dict[pp] - mu) / sigma
    return parapair_score_dict

def load_and_normalize_parapair_scores(score_filedir, norm):
    score_dict = dict()
    for fname in os.listdir(score_filedir):
        with open(score_filedir+"/"+fname, 'r') as sf:
            if norm == 'z':
                score_dict[fname] = zscore_normalize_parapair_scores(json.load(sf))
            else:
                score_dict[fname] = normalize_parapair_scores(json.load(sf))
    return score_dict

def predict_scores(score_dict, model_dict):
    assert score_dict.keys() == model_dict.keys()
    parapair_keys = score_dict[random.sample(model_dict.keys(), 1)[0]].keys()
    combined_scores = dict()
    count = 0
    for pp in parapair_keys:
        score = 0
        for fet in model_dict.keys():
            if pp in score_dict[fet].keys():
                score += score_dict[fet][pp] * model_dict[fet]
            else:
                score += score_dict[fet][rev(pp)] * model_dict[fet]
        combined_scores[pp] = score
        count += 1
        if count % 10000:
            print(".")
    min_score = min(combined_scores.values())
    print("Min score: {}".format(min_score))
    for pp in combined_scores.keys():
        combined_scores[pp] = combined_scores[pp] + abs(min_score)
    return combined_scores

def main():
    parser = argparse.ArgumentParser(description="Get combined parapair score using pre-trained model file")
    parser.add_argument("-pps", "--parapair_score_dir", required=True,
                        help="Path to directory containing parapair scores to be combined")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    parser.add_argument("-n", "--normalization", required=True,
                        help="Normalization method to be used (n = scale values between [0,1] / z = z-score normalization")
    parser.add_argument("-o", "--out", required=True, help="Path to output parapair score file")
    args = vars(parser.parse_args())
    parapair_score_dir = args["parapair_score_dir"]
    model_file = args["model"]
    norm = args["normalization"]
    out_file = args["out"]

    parapair_scores = load_and_normalize_parapair_scores(parapair_score_dir, norm)
    with open(model_file, 'r') as m:
        model = json.load(m)
    combined_score_dict = predict_scores(parapair_scores, model)
    with open(out_file, 'w') as out:
        json.dump(combined_score_dict, out)

if __name__ == '__main__':
    main()