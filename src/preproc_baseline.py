#!/usr/bin/python3

import para_preprocessor, math, json, statistics, sys, argparse
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

def get_doclen_stats(preproc_para_token_freq):
    doc_lens = dict()
    sum_len = 0
    for p in preproc_para_token_freq.keys():
        doc_len = 0
        for t in preproc_para_token_freq[p]:
            doc_len += preproc_para_token_freq[p][t]
        doc_lens[p] = doc_len
        sum_len += doc_len
    return float(sum_len/len(doc_lens.keys())), doc_lens

def bm25_similarity(q, d, para_token_freq, df, avg_doclen, doc_lens, k1=1.2, b=0.75):
    # formula used as described in the following link
    # https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables

    bm25_score = 0.0
    doc_count = len(para_token_freq.keys())
    for t in para_token_freq[q]:
        if t in para_token_freq[d].keys():
            idf_qi = math.log(1 + (doc_count - df[t] + 0.5)/(df[t] + 0.5))
            bm25_score += idf_qi * para_token_freq[d][t] * (k1 + 1) / (para_token_freq[d][t] * k1 * (1 - b + b * doc_lens[d] / avg_doclen))
    return bm25_score

def get_parapair_bm25_scores(parapair_dict, para_token_freq, df, avg_doclen, doc_lens):
    scores = dict()
    i = 0
    for page in parapair_dict.keys():
        for pp in parapair_dict[page]['parapairs']:
            p1 = pp.split("_")[0]
            p2 = pp.split("_")[1]
            sim_score = statistics.mean([bm25_similarity(p1, p2, para_token_freq, df, avg_doclen, doc_lens),
                            bm25_similarity(p2, p1, para_token_freq, df, avg_doclen, doc_lens)])
            scores[pp] = sim_score
            i += 1
            if i%10000 == 0:
                print(".")
    return scores

def odd_one_out_bm25(page, triple_data, all_terms, para_token_freq, df, avg_doclen, doc_lens):
    print("\n" + page)
    acc = 0
    count = 0
    performance = []
    triples_count = len(triple_data[()][page])
    for t in triple_data[()][page]:
        para_s1 = t[0]
        para_s2 = t[1]
        para_o = t[2]

        sim_s1s2 = statistics.mean([bm25_similarity(para_s1, para_s2, para_token_freq, df, avg_doclen, doc_lens),
                        bm25_similarity(para_s2, para_s1, para_token_freq, df, avg_doclen, doc_lens)])
        sim_s1o = statistics.mean([bm25_similarity(para_s1, para_o, para_token_freq, df, avg_doclen, doc_lens),
                        bm25_similarity(para_o, para_s1, para_token_freq, df, avg_doclen, doc_lens)])
        sim_s2o = statistics.mean([bm25_similarity(para_s2, para_o, para_token_freq, df, avg_doclen, doc_lens),
                        bm25_similarity(para_o, para_s2, para_token_freq, df, avg_doclen, doc_lens)])

        count += 1
        if sim_s1s2 > sim_s1o and sim_s1s2 > sim_s2o:
            # correct
            acc += 1
            performance.append(1)
        else:
            # incorrect
            performance.append(0)
        print(str(round((count * 100 / triples_count), 2)) + " % completion: Accuracy " + str(
            round((acc * 100 / count), 4)) + " %", end='\r')
    return performance, round((acc * 100 / count), 4)

def get_parapair_tfidf_scores(parapair_dict, para_tfidf):
    scores = dict()
    i = 0
    for page in parapair_dict.keys():
        for pp in parapair_dict[page]['parapairs']:
            p1 = pp.split("_")[0]
            p2 = pp.split("_")[1]
            p1_vec = para_tfidf[p1]
            p2_vec = para_tfidf[p2]
            sim_score = 1 - distance.cosine(p1_vec, p2_vec)
            scores[pp] = sim_score
            i += 1
            if i % 10000 == 0:
                print(".")
    return scores

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

        s1_vec = para_tfidf[para_s1]
        s2_vec = para_tfidf[para_s2]
        o_vec = para_tfidf[para_o]

        if np.count_nonzero(s1_vec) < 1 or np.count_nonzero(s2_vec) < 1 or np.count_nonzero(o_vec) < 1:
            continue
        dist_s1s2 = distance.cosine(s1_vec, s2_vec)
        dist_s1o = distance.cosine(s1_vec, o_vec)
        dist_s2o = distance.cosine(s2_vec, o_vec)

        count += 1
        if dist_s1s2 < dist_s1o and dist_s1s2 < dist_s2o:
            # correct
            acc += 1
            performance.append(1)
        else:
            # incorrect
            performance.append(0)
        print(str(round((count * 100 / triples_count), 2)) + " % completion: Accuracy " + str("%08.4f"%round((acc * 100 / count), 4)) + " %", end='\r')
    return performance, round((acc * 100 / count), 4)

def odd_one_out_precomputed_scores(page, triple_data, parapair_score_dict, out):
    print("\n" + page)
    acc = 0
    count = 0
    performance = []
    triples_count = len(triple_data[()][page])
    for t in triple_data[()][page]:
        para_s1 = t[0]
        para_s2 = t[1]
        para_o = t[2]

        if para_s1+"_"+para_s2 in parapair_score_dict.keys():
            sim_s1s2 = parapair_score_dict[para_s1+"_"+para_s2]
        elif para_s2+"_"+para_s1 in parapair_score_dict.keys():
            sim_s1s2 = parapair_score_dict[para_s2+"_"+para_s1]
        else:
            print("No score found for pair: {}".format(para_s1+" and "+para_s2))
            sim_s1s2 = 0

        if para_s1+"_"+para_o in parapair_score_dict.keys():
            sim_s1o = parapair_score_dict[para_s1+"_"+para_o]
        elif para_o+"_"+para_s1 in parapair_score_dict.keys():
            sim_s1o = parapair_score_dict[para_o+"_"+para_s1]
        else:
            print("No score found for pair: {}".format(para_s1+" and "+para_o))
            sim_s1o = 0

        if para_s2+"_"+para_o in parapair_score_dict.keys():
            sim_s2o = parapair_score_dict[para_s2+"_"+para_o]
        elif para_o+"_"+para_s2 in parapair_score_dict.keys():
            sim_s2o = parapair_score_dict[para_o+"_"+para_s2]
        else:
            print("No score found for pair: {}".format(para_s2+" and "+para_o))
            sim_s2o = 0

        count += 1
        if sim_s1s2 > sim_s1o and sim_s1s2 > sim_s2o:
            # correct
            acc += 1
            performance.append(1)
        else:
            # incorrect
            performance.append(0)
        print(str(round((count * 100 / triples_count), 2)) + " % completion: Accuracy " + str(
            "%08.4f" % round((acc * 100 / count), 4)) + " %", end='\r')
    out.write(page+" "+str(round((acc * 100 / count), 4))+"\n")
    return performance, round((acc * 100 / count), 4)

# def main():
#     page_paras_json_file = sys.argv[1]
#     triples_file = sys.argv[2]
#     preproc_para_tokens_file = sys.argv[3]
#     performance_out = sys.argv[4]
#     method = sys.argv[5]  # bm25/tfidf
#     preproc_para_token_freq = para_preprocessor.get_para_token_freq(np.load(preproc_para_tokens_file))
#     all_terms = get_terms_list(preproc_para_token_freq)
#     df = get_df(all_terms, preproc_para_token_freq)
#     avg_doclen, doc_lens = get_doclen_stats(preproc_para_token_freq)
#     triple_data = np.load(triples_file)
#     triple_performance = dict()
#     page_accuracy = dict()
#
#     with open(page_paras_json_file, 'r') as pf:
#         page_paras = json.load(pf)
#         if method == "bm25":
#             print("bm25")
#             for page in page_paras.keys():
#                 if page in triple_data[()].keys():
#                     triple_performance[page], page_accuracy[page] = odd_one_out_bm25(page, triple_data, all_terms, preproc_para_token_freq, df, avg_doclen, doc_lens)
#         else:
#             print("tfidf")
#             for page in page_paras.keys():
#                 if page in triple_data[()].keys():
#                     para_tfidf_dict = get_para_tfidf_vec_page(page_paras[page], preproc_para_token_freq, df)
#                     para_tfidf = dict()
#                     for p in para_tfidf_dict.keys():
#                         para_tfidf[p] = expand_term_vec(para_tfidf_dict[p], all_terms)
#                     triple_performance[page], page_accuracy[page] = odd_one_out(page, triple_data, para_tfidf)
#     np.save(performance_out, triple_performance)
#     mean_acc = 0
#     for p in page_accuracy.keys():
#         mean_acc += page_accuracy[p]
#     mean_acc = mean_acc / len(page_accuracy)
#     stderr = statistics.stdev(page_accuracy.values())/np.sqrt(len(page_accuracy))
#     print("\n\nMean triple accuracy: p p "+str(round(mean_acc, 4))+" %")
#     print("stderr: p p p p "+str(round(stderr, 4)))

def main():
    parser = argparse.ArgumentParser(description="Calculate odd-one-out performance based on parapair scores")
    parser.add_argument("-pgp", "--page_paras", required=True, help="Path to page paras json file")
    parser.add_argument("-trp", "--triples", required=True, help="Path to triples dataset (numpy format)")
    parser.add_argument("-pps", "--parapair_score", required=True, help="Path to parapair score file")
    parser.add_argument("-o", "--out", required=True, help="Path to output log file")
    parser.add_argument("-m", "--method", required=True, help="Method name")
    parser.add_argument("-po", "--performance_out", help="Path to detailed performance out put file saved as np array")
    args = vars(parser.parse_args())
    page_paras_json_file = args["page_paras"]
    triples_file = args["triples"]
    parapair_score_file = args["parapair_score"]
    out_log_file = args["out"]
    m_name = args["method"]
    performance_out = args["performance_out"]

    with open(parapair_score_file, 'r') as pps:
        parapair_score_dict = json.load(pps)

    triple_data = np.load(triples_file)
    triple_performance = dict()
    page_accuracy = dict()
    print("Going to calculate odd-one-out")
    log = open(out_log_file, 'w')
    log.write(m_name+" Accuracy\n")
    with open(page_paras_json_file, 'r') as pf:
        page_paras = json.load(pf)
        for page in page_paras.keys():
            if page in triple_data[()].keys():
                triple_performance[page], page_accuracy[page] = odd_one_out_precomputed_scores(page, triple_data, parapair_score_dict, log)
    np.save(performance_out, triple_performance)
    mean_acc = 0
    for p in page_accuracy.keys():
        mean_acc += page_accuracy[p]
    mean_acc = mean_acc / len(page_accuracy)
    stderr = statistics.stdev(page_accuracy.values()) / np.sqrt(len(page_accuracy))
    print("\n\nMean triple accuracy: p p " + str(round(mean_acc, 4)) + " %")
    print("stderr: p p p p " + str(round(stderr, 4)))
    log.write("\nMean "+str(round(mean_acc, 4))+"\n")
    log.write("stderr "+str(round(stderr, 4)))
    log.close()

if __name__ == '__main__':
    main()