#!/usr/bin/python3

import para_preprocessor, math, json
import numpy as np
from scipy.spatial import distance

def get_terms_list(all_para_tfs):
    all_terms = set()
    for p in all_para_tfs.keys():
        all_terms = all_terms.union(all_para_tfs[p].keys())
    return list(all_terms)

def get_df(all_terms, all_para_tfs):
    df = dict()
    for t in all_terms:
        df[t] = 0
        for p in all_para_tfs.keys():
            if t in all_para_tfs[p].keys():
                df[t] += 1
    return df

def expand_term_vec(term_vec, all_terms):
    expanded_term_vec = []
    for i in range(len(all_terms)):
        t = all_terms[i]
        if t in term_vec.keys():
            expanded_term_vec.append(term_vec[t])
        else:
            expanded_term_vec.append(0)
    return np.array(expanded_term_vec)

def get_para_tfidf_vec_page(paras_in_page, all_para_tfs, df):
    para_tfidf = dict()
    n = len(all_para_tfs.keys())
    for para in paras_in_page:
        tf = all_para_tfs[para]
        doclen = sum(tf.values())
        tfidf_vec = dict()
        for t in tf.keys():
            tfidf_vec[t] = (tf[t]/doclen) * (np.log(n)-np.log(df[t]))
        para_tfidf[para] = tfidf_vec
    return para_tfidf

def odd_one_out(page, triple_data, para_tfidf):
    print("\n"+page)
    acc = 0
    count = 0
    performance = []
    triples_count = len(triple_data[()][page])
    for t in triple_data[()][page]:
        para_s1 = t[0]
        para_s2 = t[1]
        para_o = t[2]

        dist_s1s2 = distance.cosine(para_tfidf[para_s1], para_tfidf[para_s2])
        dist_s1o = distance.cosine(para_tfidf[para_s1], para_tfidf[para_o])
        dist_s2o = distance.cosine(para_tfidf[para_s2], para_tfidf[para_o])

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

def main():
    page_paras_json = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.page.paras.json"
    paratext_json = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.para.texts.json"
    triples_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1-train-nodup.triples.npy"
    preproc_para_tokens_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup-preprocessed-para-token-dict/by1train_paras_preproc.npy"
    # preproc_para_tokens = para_preprocessor.preprocess_paras(paratext_json, stemlem)
    # np.save("/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup-preprocessed-para-token-dict/by1train_paras_preproc", preproc_para_tokens)
    performance_out = "/home/sumanta/Documents/Dugtrio-data/Odd-One-Out/by1train-nodup-preproc-triples-performance/by1train-nodup-tfidf.triple.performance"
    preproc_para_token_freq = para_preprocessor.get_para_token_freq(np.load(preproc_para_tokens_file))
    all_terms = get_terms_list(preproc_para_token_freq)
    df = get_df(all_terms, preproc_para_token_freq)
    triple_data = np.load(triples_file)
    triple_performance = dict()
    page_accuracy = dict()
    with open(page_paras_json, 'r') as pf:
        page_paras = json.load(pf)
        for page in page_paras.keys():
            para_tfidf_dict = get_para_tfidf_vec_page(page_paras[page], preproc_para_token_freq, df)
            para_tfidf = dict()
            for p in para_tfidf_dict.keys():
                para_tfidf[p] = expand_term_vec(para_tfidf_dict[p], all_terms)
            triple_performance[page], page_accuracy[page] = odd_one_out(page, triple_data, para_tfidf)
    np.save(performance_out, triple_performance)
    mean_acc = 0
    for p in page_accuracy.keys():
        mean_acc += page_accuracy[p]
    mean_acc = mean_acc / len(page_accuracy)
    print("\n\nMean triple accuracy: "+str(round(mean_acc, 4))+" %")

if __name__ == '__main__':
    main()