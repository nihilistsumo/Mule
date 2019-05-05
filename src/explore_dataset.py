#!/usr/bin/python3
import json, sys, math, time
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial


def load_data():
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
    with open(terms_freq_file, 'r') as gtf:
        global_term_freq = json.load(gtf)
    with open(terms_doc_freq_file, 'r') as gdf:
        global_term_doc_freq = json.load(gdf)

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

    return per_label_freq_vec, terms, labels, page_paras, page_tops, inds, freqs, global_term_freq, global_term_doc_freq

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

def calc_para_lmjm(page):
    cum_page_freq = np.zeros(len(terms))
    for k in page_freq_vec[page].keys():
        cum_page_freq += page_freq_vec[page][k]
    c_term_freq = np.array(term_freq - cum_page_freq)

    paras = page_paras[page]
    para_lmjm = dict()
    para_term_freq_vecs = dict()
    for p in paras:
        pind = para_inds[p]
        pfreq = para_freqs[p]
        p_term_vec = np.zeros(len(terms))
        for i in range(len(pind)):
            p_term_vec[pind[i]] = pfreq[i]
        para_term_freq_vecs[p] = p_term_vec
        para_lmjm[p] = jm_smoothing(p_term_vec, c_term_freq)
    return  para_lmjm, para_term_freq_vecs, c_term_freq

def explore_label_dist(page):
    page_para_lmjm, page_para_term_freq_vecs, page_c_term_freq = calc_para_lmjm(page)
    label_lmjm = []
    for l in page_freq_vec[page].keys():
        label_lmjm.append(jm_smoothing(page_freq_vec[page][l], page_c_term_freq))
    label_lmjm = np.array(label_lmjm)
    plt.imshow(label_lmjm, cmap='hot', interpolation='nearest')
    plt.show()

terms_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.global.terms.json"
terms_freq_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.global.terms.freq.json"
terms_doc_freq_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.global.terms.doc.freq.json"
labels_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.page.para.labels.json"
page_paras_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.page.paras.json"
page_topics_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.page.tops.json"
para_freq_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.para.freqs.json"
para_ind_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.para.indices.json"
triples_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1-train-nodup.triples.npy"

page_freq_vec, terms, page_para_labels, page_paras, page_tops, para_inds, para_freqs, term_freq, term_doc_freq = load_data()
page = "enwiki:Subprime%20mortgage%20crisis"
explore_label_dist(page)