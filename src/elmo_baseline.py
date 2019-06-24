#!/usr/bin/python3

import math, json, os, sys, argparse
import numpy as np
from scipy.spatial import distance

def get_page_para_elmo_vecs(page, elmo_data):
    page_data = elmo_data[()][page]
    paraids = page_data['paraids']
    para_sent_count = page_data['para_sent_count']
    para_sent_vecs = page_data['sent_vecs']
    para_elmo_sent_vecs = dict()
    i = 0
    for j in range(len(paraids)):
        paraid = paraids[j]
        para_vecs = []
        for k in range(para_sent_count[j]):
            para_vecs.append(para_sent_vecs[i])
            i += 1
        para_elmo_sent_vecs[paraid] = para_vecs
    return para_elmo_sent_vecs

def get_dist_first_sent(para_s1_vecs, para_s2_vecs, para_o_vecs):
    dist_s1s2 = distance.cosine(para_s1_vecs[0], para_s2_vecs[0])
    dist_s1o = distance.cosine(para_s1_vecs[0], para_o_vecs[0])
    dist_s2o = distance.cosine(para_s2_vecs[0], para_o_vecs[0])
    return dist_s1s2, dist_s1o, dist_s2o

def get_dist_min_link(para_s1_vecs, para_s2_vecs, para_o_vecs):
    dist_s1s2 = sys.float_info.max
    for i in range(len(para_s1_vecs)):
        for j in range(len(para_s2_vecs)):
            d = distance.cosine(para_s1_vecs[i], para_s2_vecs[j])
            if d < dist_s1s2:
                dist_s1s2 = d

    dist_s1o = sys.float_info.max
    for i in range(len(para_s1_vecs)):
        for j in range(len(para_o_vecs)):
            d = distance.cosine(para_s1_vecs[i], para_o_vecs[j])
            if d < dist_s1o:
                dist_s1o = d

    dist_s2o = sys.float_info.max
    for i in range(len(para_s2_vecs)):
        for j in range(len(para_o_vecs)):
            d = distance.cosine(para_s2_vecs[i], para_o_vecs[j])
            if d < dist_s2o:
                dist_s2o = d
    return dist_s1s2, dist_s1o, dist_s2o

def get_dist_avg_link(para_s1_vecs, para_s2_vecs, para_o_vecs):
    para_s1_avg_vec = np.mean(para_s1_vecs, axis=0)
    para_s2_avg_vec = np.mean(para_s2_vecs, axis=0)
    para_o_avg_vec = np.mean(para_o_vecs, axis=0)
    dist_s1s2 = distance.cosine(para_s1_avg_vec, para_s2_avg_vec)
    dist_s1o = distance.cosine(para_s1_avg_vec, para_o_avg_vec)
    dist_s2o = distance.cosine(para_s2_avg_vec, para_o_avg_vec)
    return dist_s1s2, dist_s1o, dist_s2o

def get_parapair_elmo_scores(parapair_dict, elmo_data, method):
    scores = dict()
    i = 0
    for page in parapair_dict.keys():
        for pp in parapair_dict[page]['parapairs']:
            p1 = pp.split("_")[0]
            p2 = pp.split("_")[1]
            p1_vecs = elmo_data[()][p1]
            p2_vecs = elmo_data[()][p2]
            if method == 'first':
                sim_score = 1 - distance.cosine(p1_vecs[0], p2_vecs[0])
            elif method == 'min':
                dist = sys.float_info.max
                for i in range(len(p1_vecs)):
                    for j in range(len(p2_vecs)):
                        d = distance.cosine(p1_vecs[i], p2_vecs[j])
                        if d < dist:
                            dist = d
                sim_score = 1 - dist
            else:
                p1_avg_vec = np.mean(p1_vecs, axis=0)
                p2_avg_vec = np.mean(p2_vecs, axis=0)
                sim_score = 1 - distance.cosine(p1_avg_vec, p2_avg_vec)
            scores[pp] = sim_score
            i += 1
            if i % 10000 == 0:
                print(".")
    return scores

def odd_one_out(page, triple_data, para_elmo_vecs, method):
    print("\n"+page)
    acc = 0
    count = 0
    performance = []
    triples_count = len(triple_data[()][page])
    for t in triple_data[()][page]:
        para_s1 = t[0]
        para_s2 = t[1]
        para_o = t[2]

        if len(para_elmo_vecs[para_s1]) < 1:
            print(para_s1+" has no vec associated")
            return performance, 0.0
        elif len(para_elmo_vecs[para_s2]) < 1:
            print(para_s2+" has no vec associated")
            return performance, 0.0
        elif len(para_elmo_vecs[para_o]) < 1:
            print(para_o+" has no vec associated")
            return performance, 0.0
        else:
            para_s1_vecs = para_elmo_vecs[para_s1]
            para_s2_vecs = para_elmo_vecs[para_s2]
            para_o_vecs = para_elmo_vecs[para_o]

        if method == 'first':
            dist_s1s2, dist_s1o, dist_s2o = get_dist_first_sent(para_s1_vecs, para_s2_vecs, para_o_vecs)
        elif method == 'min':
            dist_s1s2, dist_s1o, dist_s2o = get_dist_min_link(para_s1_vecs, para_s2_vecs, para_o_vecs)
        else:
            dist_s1s2, dist_s1o, dist_s2o = get_dist_avg_link(para_s1_vecs, para_s2_vecs, para_o_vecs)

        count += 1
        if dist_s1s2 < dist_s1o and dist_s1s2 < dist_s2o:
            # correct
            acc += 1
            performance.append(1)
        else:
            # incorrect
            performance.append(0)
        print(str(round((count * 100 / triples_count), 2)) + " % completion: Accuracy " + str(round((acc * 100 / count), 4)) + " %", end='\r')
    return performance, round((acc * 100 / count), 4)

# def main():
#     elmo_data_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1test-nodup-elmo-vec-data/by1test_merged_elmo_data_squeezed.npy"
#     page_paras_json = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1test-nodup.json.data/by1-test-nodup.page.paras.json"
#     triples_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1-test-nodup.triples.npy"
#     #performance_out = "/home/sumanta/Documents/Dugtrio-data/Odd-One-Out/by1train-nodup-preproc-triples-performance/by1test-nodup-tfidf.triple.performance"
#     triple_data = np.load(triples_file)
#     triple_performance = dict()
#     page_accuracy = dict()
#     elmo_data = np.load(elmo_data_file)
#     with open(page_paras_json, 'r') as pf:
#         page_paras = json.load(pf)
#         for page in page_paras.keys():
#             para_elmo_vecs = get_page_para_elmo_vecs(page, elmo_data)
#             triple_performance[page], page_accuracy[page] = odd_one_out(page, triple_data, para_elmo_vecs, '')
#     # np.save(performance_out, triple_performance)
#     mean_acc = 0
#     for p in page_accuracy.keys():
#         mean_acc += page_accuracy[p]
#     mean_acc = mean_acc / len(page_accuracy)
#     print("\n\nMean triple accuracy: "+str(round(mean_acc, 4))+" %")

def main():
    parser = argparse.ArgumentParser(description="Calculate parapair scores and save them in a dict")
    parser.add_argument("-e", "--elmo_data", required=True, help="Path to ELMo data file")
    parser.add_argument("-pp", "--parapairs", required=True, help="Path to parapairs file")
    parser.add_argument("-m", "--method", required=True, help="Method name (first/min/avg)")
    parser.add_argument("-o", "--out", required=True, help="Path to output file")
    args = vars(parser.parse_args())
    elmo_data_file = args["elmo_data"]
    parapairs_file = args["parapairs"]
    method = args["method"]
    outfile = args["out"]
    with open(parapairs_file, 'r') as ppf:
        pp_data = json.load(ppf)
    # parapairs = pp_data['parapairs']
    elmo_data = np.load(elmo_data_file)
    parapair_scores = get_parapair_elmo_scores(pp_data, elmo_data, method)
    with open(outfile, 'w') as out:
        json.dump(parapair_scores, out)

if __name__ == '__main__':
    main()