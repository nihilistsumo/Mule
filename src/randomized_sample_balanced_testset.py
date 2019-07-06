import numpy as np
import tensorflow as tf
import json, os, argparse, random
from scipy import stats
from collections import Counter

def get_samples(parapair_data):
    pairs = parapair_data["parapairs"]
    labels = parapair_data["labels"]
    pos_pairs = []
    neg_pairs = []
    for i in range(len(labels)):
        if labels[i] == 0:
            neg_pairs.append(pairs[i])
        else:
            pos_pairs.append(pairs[i])
    return pos_pairs, neg_pairs

def get_discriminative_samples(parapair_data, hier_qrels_reverse_dict):
    pairs = parapair_data["parapairs"]
    labels = parapair_data["labels"]
    pos_pairs = []
    neg_pairs = []
    for i in range(len(labels)):
        if labels[i] == 0:
            neg_pairs.append(pairs[i])
        else:
            p1 = pairs[i].split("_")[0]
            p2 = pairs[i].split("_")[1]
            if hier_qrels_reverse_dict[p1] == hier_qrels_reverse_dict[p2]:
                pos_pairs.append(pairs[i])
    return pos_pairs, neg_pairs

def sample_randomized_balanced(parapair_dict, hier_qrels_reverse, discrim):
    balanced_dict = dict()
    for page in parapair_dict.keys():
        balanced_dict[page] = dict()
        pairs = []
        labels = []
        if discrim:
            test_pos, test_neg = get_discriminative_samples(parapair_dict[page], hier_qrels_reverse)
        else:
            test_pos, test_neg = get_samples(parapair_dict[page])
        test_neg = random.sample(test_neg, len(test_pos))
        for p in test_pos:
            pairs.append(p)
            labels.append(1)
        for p in test_neg:
            pairs.append(p)
            labels.append(0)
        balanced_dict[page]["parapairs"] = pairs
        balanced_dict[page]["labels"] = labels
        print(page)
    return balanced_dict

def main():
    parser = argparse.ArgumentParser(description="Create a balanced test dataset out of all test samples")
    parser.add_argument("-pp", "--parapair", required=True, help="Path to parapair file")
    parser.add_argument("-hq", "--hier_qrels", required=True, help="Path to hierarchical qrels file")
    parser.add_argument("-d", "--discriminative", required=True, help="Discriminative (d) / normal (n) dataset")
    parser.add_argument("-o", "--out", required=True, help="Path to output file")
    args = vars(parser.parse_args())
    parapair_file = args["parapair"]
    hier_qrels_file = args["hier_qrels"]
    discrim = args["discriminative"]
    outfile = args["out"]

    with open(parapair_file, 'r') as pp:
        parapair = json.load(pp)

    hier_qrels_reverse = dict()
    with open(hier_qrels_file, 'r') as hq:
        for l in hq:
            hier_qrels_reverse[l.split(" ")[2]] = l.split(" ")[0]

    balanced_test = sample_randomized_balanced(parapair, hier_qrels_reverse, discrim=="d")

    with open(outfile, 'w') as out:
        json.dump(balanced_test, out)

if __name__ == '__main__':
    main()