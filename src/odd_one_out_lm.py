#!/usr/bin/python3
import json, sys, math
import numpy as np
from scipy import special
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def get_page_term_freq_vecs(page_paras_file, page_topics_file, labels_file, terms_file, para_ind_file, para_freq_file):
    with open(terms_file,'r') as tf:
        terms = json.load(tf)
    with open(labels_file,'r') as lf:
        labels = json.load(lf)
    with open(page_paras_file,'r') as pf:
        page_paras = json.load(pf)
    with open(page_topics_file,'r') as tf:
        page_tops = json.load(tf)
    with open(para_freq_file,'r') as ff:
        freqs = json.load(ff)
    with open(para_ind_file,'r') as indf:
        inds = json.load(indf)

    per_label_freq_vec = dict()
    for page in page_paras:
        print(page)
        labels_in_page = labels[page]
        label_term_freq_page = dict()
        for l in set(labels_in_page.values()):
            label_term_freq_page[l] = np.zeros(len(terms))

        for p in page_paras[page]:
            label = labels_in_page[p]
            for i in range(len(inds[p])):
                label_term_freq_page[label][inds[p][i]] += freqs[p][i]

        per_label_freq_vec[page] = label_term_freq_page

    return per_label_freq_vec, terms, labels, page_paras, page_tops, inds, freqs

def jm_smoothing(vec, c_vec, lamb=0.9):
    smoothed_vec = np.zeros(len(vec))
    d = np.sum(vec)
    C = np.sum(c_vec)
    if d<1:
        smoothed_vec = c_vec / C
    else:
        for i in range(len(vec)):
            smoothed_vec[i] = lamb*vec[i]/d + (1-lamb)*c_vec[i]/C
    return smoothed_vec

def kldiv(p,q):
    _big_number = 1000.0
    if len(p) != len(q):
        raise Exception("Can not calculate KLdiv of two different length vectors: "+str(len(p))+","+str(len(q)))
    score = 0
    for x in range(len(p)):
        if p[x] - q[x] < sys.float_info.min:
            continue
        # elif p[x] < sys.float_info.min:
        #     score += _big_number * q[x]
        # elif q[x] < sys.float_info.min:
        #     score += _big_number * p[x]
        elif p[x] < sys.float_info.min or q[x] < sys.float_info.min:
            score += _big_number
        else:
            score += p[x] * math.log(p[x]/q[x])
    return score

def calc_accuracy(page):
    print(page+" started")
    cum_page_freq = np.zeros(len(terms))
    for k in page_freq_vec[page].keys():
        cum_page_freq += page_freq_vec[page][k]
    c_term_freq = np.array(global_term_freq - cum_page_freq)
    acc = 0.0
    for t in triples_data[()][page]:
        top_freq_vec_page = page_freq_vec[page]
        p_term_vecs = []
        for p in t:
            plabel = page_para_labels[page][p]
            pind = para_inds[p]
            pfreq = para_freqs[p]
            p_term_vec = np.zeros(len(terms))
            for i in range(len(pind)):
                p_term_vec[pind[i]] = pfreq[i]
            p_term_vecs.append(p_term_vec)
            top_freq_vec_page[plabel] -= p_term_vec

        lm_in_page = []
        for l in top_freq_vec_page.keys():
            lm_in_page.append(jm_smoothing(top_freq_vec_page[l], c_term_freq))
        lm_in_page = np.array(lm_in_page)
        for i in range(3):
            p_term_vecs[i] = jm_smoothing(p_term_vecs[i], c_term_freq)

        kldiv_vecs = []
        for i in range(3):
            kldiv_vec = []
            for l in range(len(lm_in_page)):
                kldiv_vec.append(kldiv(p_term_vecs[i], lm_in_page[l]))
            kldiv_vec = np.array(kldiv_vec)
            kldiv_vec = kldiv_vec / np.sum(kldiv_vec)
            kldiv_vecs.append(kldiv_vec)
            # print(str(i+1)+": "+t[i]+": "+str(kldiv_vec.tolist()))
        dist_p12 = kldiv(kldiv_vecs[0], kldiv_vecs[1])
        dist_p23 = kldiv(kldiv_vecs[1], kldiv_vecs[2])
        dist_p31 = kldiv(kldiv_vecs[2], kldiv_vecs[0])
        if (dist_p12 < dist_p23 and dist_p12 < dist_p31):
            acc += 1
        # print("Similar paras: ("+t[0]+","+t[1]+"), Odd: "+t[2])
        # print("1,2: "+str(kldiv(kldiv_vecs[0],kldiv_vecs[1]))+", 2,3: "+str(kldiv(kldiv_vecs[1],kldiv_vecs[2]))+
        #       ", 3,1: "+str(kldiv(kldiv_vecs[2],kldiv_vecs[0])))
    accuracy = acc * 100 / len(triples_data[()][page])
    print(page + " Accuracy: " + accuracy + "%")


# terms_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.global.terms.json"
# terms_freq_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.global.terms.freq.json"
# labels_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.page.para.labels.json"
# page_paras_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.page.paras.json"
# page_topics_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.page.tops.json"
# para_freq_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.para.freqs.json"
# para_ind_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.para.indices.json"
# triples_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1-train-nodup.triples.npy"

terms_file = sys.argv[1]
terms_freq_file = sys.argv[2]
labels_file = sys.argv[3]
page_paras_file = sys.argv[4]
page_topics_file = sys.argv[5]
para_freq_file = sys.argv[6]
para_ind_file = sys.argv[7]
triples_file = sys.argv[8]
num_workers = int(sys.argv[9])

page_freq_vec, terms, page_para_labels, page_paras, page_tops, para_inds, para_freqs = \
    get_page_term_freq_vecs(page_paras_file, page_topics_file, labels_file, terms_file, para_ind_file, para_freq_file)

# triples_data = dict()
# with open(triples_file,'r') as tf:
#     for l in tf:
#         page = l.split(" ")[0]
#         #top_freq_vec_page = page_freq_vec[page]
#         t_paras = [page_paras[page][int(l.split(" ")[1])], page_paras[page][int(l.split(" ")[2])], page_paras[page][int(l.split(" ")[3])]]
#         if page in triples_data.keys():
#             triples_data[page].append(t_paras)
#         else:
#             triples_data[page] = [t_paras]
# np.save("/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1-train-nodup.triples",triples_data)

triples_data = np.load(triples_file)

with open(terms_freq_file,'r') as gtf:
    global_term_freq = json.load(gtf)
print("Data loaded")



# for page in triples_data[()].keys():
#     calc_accuracy(page)


with ProcessPoolExecutor(max_workers=num_workers) as ex:
    for page in triples_data[()].keys():
        ex.submit(calc_accuracy, page)







    # cum_page_freq = np.zeros(len(terms))
    # for k in page_freq_vec[page].keys():
    #     cum_page_freq += page_freq_vec[page][k]
    # c_term_freq = np.array(global_term_freq - cum_page_freq)
    # acc = 0.0
    # for t in triples_data[()][page]:
    #     top_freq_vec_page = page_freq_vec[page]
    #     p_term_vecs = []
    #     for p in t:
    #         plabel = page_para_labels[page][p]
    #         pind = para_inds[p]
    #         pfreq = para_freqs[p]
    #         p_term_vec = np.zeros(len(terms))
    #         for i in range(len(pind)):
    #             p_term_vec[pind[i]] = pfreq[i]
    #         p_term_vecs.append(p_term_vec)
    #         top_freq_vec_page[plabel] -= p_term_vec
    #
    #     lm_in_page = []
    #     for l in top_freq_vec_page.keys():
    #         lm_in_page.append(jm_smoothing(top_freq_vec_page[l], c_term_freq))
    #     lm_in_page = np.array(lm_in_page)
    #     for i in range(3):
    #         p_term_vecs[i] = jm_smoothing(p_term_vecs[i], c_term_freq)
    #
    #     kldiv_vecs = []
    #     for i in range(3):
    #         kldiv_vec = []
    #         for l in range(len(lm_in_page)):
    #             kldiv_vec.append(kldiv(p_term_vecs[i], lm_in_page[l]))
    #         kldiv_vec = np.array(kldiv_vec)
    #         kldiv_vec = kldiv_vec/np.sum(kldiv_vec)
    #         kldiv_vecs.append(kldiv_vec)
    #         #print(str(i+1)+": "+t[i]+": "+str(kldiv_vec.tolist()))
    #     dist_p12 = kldiv(kldiv_vecs[0],kldiv_vecs[1])
    #     dist_p23 = kldiv(kldiv_vecs[1],kldiv_vecs[2])
    #     dist_p31 = kldiv(kldiv_vecs[2],kldiv_vecs[0])
    #     if(dist_p12<dist_p23 and dist_p12<dist_p31):
    #         acc += 1
    #     # print("Similar paras: ("+t[0]+","+t[1]+"), Odd: "+t[2])
    #     # print("1,2: "+str(kldiv(kldiv_vecs[0],kldiv_vecs[1]))+", 2,3: "+str(kldiv(kldiv_vecs[1],kldiv_vecs[2]))+
    #     #       ", 3,1: "+str(kldiv(kldiv_vecs[2],kldiv_vecs[0])))
    # print(page+" Accuracy: "+acc*100/len(triples_data[()][page]))
