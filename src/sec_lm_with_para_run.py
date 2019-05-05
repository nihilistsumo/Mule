#!/usr/bin/python3
import numpy as np
import json, sys
from scipy import special, spatial
import para_preprocessor

def get_cumul_freq_dist_sec(page, sec, topsec_para, para_token_freq):
    sec_cum_freq = dict()
    sec_ret_paras = topsec_para[()][page][sec]
    for p in sec_ret_paras:
        p_term_freq = para_token_freq[()][p]
        for t in p_term_freq.keys():
            if t not in sec_cum_freq.keys():
                sec_cum_freq[t] = p_term_freq[t]
            else:
                sec_cum_freq[t] += p_term_freq[t]
    return sec_cum_freq

def get_cumul_freq_dist_page(page, page_paras, para_token_freq):
    page_cum_freq = dict()
    for para in page_paras[page]:
        for t in para_token_freq[()][para].keys():
            if t not in page_cum_freq.keys():
                page_cum_freq[t] = para_token_freq[()][para][t]
            else:
                page_cum_freq[t] += para_token_freq[()][para][t]
    return page_cum_freq

def get_collection_freq_dist(global_freq_dist, page_freq_dist):
    col_freq_dist = global_freq_dist
    absent_term_val = 0.001
    for t in col_freq_dist.keys():
        if t in page_freq_dist.keys():
            if col_freq_dist[t] == page_freq_dist[t]:
                # this is the case when the current term occurred only in this page
                # and not in any other pages
                col_freq_dist[t] = absent_term_val
            else:
                col_freq_dist[t] -= page_freq_dist[t]
    return col_freq_dist

def get_cumul_global_freq_dict(para_token_freq):
    global_freq_dist = dict()
    for p in para_token_freq[()].keys():
        for t in para_token_freq[()][p]:
            if t not in global_freq_dist.keys():
                global_freq_dist[t] = para_token_freq[()][p][t]
            else:
                global_freq_dist[t] += para_token_freq[()][p][t]
    return global_freq_dist

def get_smoothed_lmjm(all_terms_list, term_dict, collection_term_dict, lamb=0.9):
    lm_dist = np.zeros(len(all_terms_list))
    collection_len = sum(collection_term_dict.values())
    doc_len = sum(term_dict.values())
    for i in range(len(all_terms_list)):
        t = all_terms_list[i]
        if t not in term_dict.keys():
            lm_dist[i] = (1 - lamb) * collection_term_dict[t] / collection_len
        else:
            lm_dist[i] = lamb * term_dict[t] / doc_len + (1 - lamb) * collection_term_dict[t] / collection_len
    return lm_dist

def get_smoothed_lmds(all_terms_list, term_dict, collection_term_dict, mu=1):
    lm_dist = np.zeros(len(all_terms_list))
    collection_len = sum(collection_term_dict.values())
    doc_len = sum(term_dict.values())
    for i in range(len(all_terms_list)):
        t = all_terms_list[i]
        if t not in term_dict.keys():
            col_term_val = collection_term_dict[t]
            val = mu * (collection_term_dict[t]/collection_len) / (doc_len + mu)
            if val < sys.float_info.min:
                print("val is 0")
            lm_dist[i] = val
        else:
            term_val = term_dict[t]
            col_term_val = collection_term_dict[t]
            val = (term_dict[t] + mu * (collection_term_dict[t]/collection_len)) / (doc_len + mu)
            if val < sys.float_info.min:
                print("val is 0")
            lm_dist[i] = val
    return lm_dist

def page_lm(page, page_paras, topsec_para, para_token_freq, global_freq_dist, smoothing):
    #print(page+" lm analysis")
    page_freq_dist = get_cumul_freq_dist_page(page, page_paras, para_token_freq)
    collection_freq_dist = get_collection_freq_dist(global_freq_dist, page_freq_dist)
    page_sec_freq_dist = dict()
    for sec in topsec_para[()][page].keys():
        page_sec_freq_dist[sec] = get_cumul_freq_dist_sec(page, sec, topsec_para, para_token_freq)
    page_para_freq_dist = dict()
    for para in page_paras[page]:
        page_para_freq_dist[para] = para_token_freq[()][para]
    # 1. page_freq_dist: cumulative term freq dict of all paras inside page
    # 2. collection_freq_dist: global term freq dist minus the current page freq dist
    # 3. page_sec_freq_dist: dict of sections in the page to its cumulative term freq
    # of top k paras retrieved for that section according to the sec-para run used
    # 4. page_para_freq_dist: dict of paras in the page to its term freq
    all_page_terms = set()
    # for dist in page_sec_freq_dist.values():
    #     unified_page_terms = unified_page_terms.union(dist.keys())
    for dist in page_para_freq_dist.values():
        all_page_terms = all_page_terms.union(dist.keys())
    all_page_terms = list(all_page_terms)
    para_lms = dict()
    sec_lms = dict()
    if smoothing == 'ds':
        for para in page_paras[page]:
            para_lms[para] = get_smoothed_lmds(all_page_terms, page_para_freq_dist[para], collection_freq_dist)
        for sec in topsec_para[()][page].keys():
            sec_lms[sec] = get_smoothed_lmds(all_page_terms, page_sec_freq_dist[sec], collection_freq_dist)
    else:
        for para in page_paras[page]:
            para_lms[para] = get_smoothed_lmjm(all_page_terms, page_para_freq_dist[para], collection_freq_dist)
        for sec in topsec_para[()][page].keys():
            sec_lms[sec] = get_smoothed_lmjm(all_page_terms, page_sec_freq_dist[sec], collection_freq_dist)
    return np.array(all_page_terms), para_lms, sec_lms

def get_para_rep_vec(para_lm, sec_lms, dist_metric='kldiv'):
    para_rep_vec = np.zeros(len(sec_lms))
    if dist_metric == 'kldiv':
        for i in range(len(sec_lms)):
            para_rep_vec[i] = np.sum(special.kl_div(sec_lms[i], para_lm))
    elif dist_metric == 'cos':
        for i in range(len(sec_lms)):
            para_rep_vec[i] = spatial.distance.cosine(sec_lms[i], para_lm)
    return para_rep_vec

def odd_one_out(page, triple_data, page_lms, page_paras, dist_met):
    print("\n"+page)
    acc = 0
    count = 0
    performance = []
    triples_count = len(triple_data[()][page])
    sec_lms_in_page = []
    for sec in page_lms[()][page][2].keys():
        sec_lms_in_page.append(page_lms[()][page][2][sec])
    para_reps = dict()
    for para in page_paras[page]:
        para_lm = page_lms[()][page][1][para]
        para_reps[para] = get_para_rep_vec(para_lm, sec_lms_in_page, dist_met)
    for t in triple_data[()][page]:
        para_s1 = t[0]
        para_s2 = t[1]
        para_o = t[2]

        dist_s1s2 = spatial.distance.cosine(para_reps[para_s1], para_reps[para_s2])
        dist_s1o = spatial.distance.cosine(para_reps[para_s1], para_reps[para_o])
        dist_s2o = spatial.distance.cosine(para_reps[para_s2], para_reps[para_o])

        count += 1
        if dist_s1s2 < dist_s1o and dist_s1s2 < dist_s2o:
            # correct
            acc += 1
            performance.append(1)
        else:
            # incorrect
            performance.append(0)
        print(str(round((count * 100 / triples_count), 2)) + " % completion: Accuracy " + str(
            round((acc * 100 / count), 4)) + " %", end='\r')
    return performance


def main():
    page_paras_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.page.paras.json"
    sec_ret_topk_para_file = "/home/sumanta/Documents/Dugtrio-data/Odd-One-Out/by1train_candidate_para_runs/ghetto_sdm_top100_paragraph_page_sec.npy"
    para_token_freq_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup-preprocessed-para-token-dict/tf_dict/by1train_cand_top100+page_paras_preproc_tf.npy"
    output_data_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup-preprocessed-lm-data/by1train-nodup-lem-lmds-data.npy"
    triples_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1-train-nodup.triples.npy"

    with open(page_paras_file, 'r') as pp:
        page_paras = json.load(pp)

    # topsec_para = np.load(sec_ret_topk_para_file)
    # para_token_freq = np.load(para_token_freq_file)
    # page_lms = dict()
    # for page in page_paras.keys():
    #     print(page)
    #     terms_list, para_lms, sec_lms = page_lm(page, page_paras, topsec_para, para_token_freq, get_cumul_global_freq_dict(para_token_freq), 'jm')
    #     page_lms[page] = (terms_list, para_lms, sec_lms)
    # #np.save("/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup-preprocessed-lm-data/by1train-nodup-pureterm-lmjm-data", np.array(page_lms))
    # np.save(output_data_file, np.array(page_lms))

    variation = output_data_file.split("/")[len(output_data_file.split("/"))-1]
    print("Page LM variation: "+variation)
    triple_data = np.load(triples_file)
    page_lms = np.load(output_data_file)
    triples_performance = dict()
    for page in page_paras.keys():
        triples_performance[page] = odd_one_out(page, triple_data, page_lms, page_paras, 'cos')
    #np.save("/home/sumanta/Documents/Dugtrio-data/Odd-One-Out/by1train-nodup-preproc-triples-performance/"+variation+".triple.performance", np.array(triples_performance))

if __name__ == '__main__':
    main()