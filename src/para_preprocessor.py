#!/usr/bin/python3
import json, sys, math, time, re, string
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import special
import lucene
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from java.nio.file import Path, Paths
from org.apache.lucene.index import IndexWriter, IndexReader
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.queryparser.classic import QueryParser
#from org.apache.lucene.store import FSDirectory
from org.apache.lucene.store import MMapDirectory, SimpleFSDirectory
from org.apache.lucene.index import DirectoryReader

def init_lucene(dir_path):
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    store = SimpleFSDirectory(Paths.get(dir_path))
    searcher = IndexSearcher(DirectoryReader.open(store))
    # store.close()
    return searcher
    # IndexReader
    # ir = DirectoryReader.open(FSDirectory.open((new File(indexDirWithTermVecPath)).toPath()));
    # IndexSearcher is = new
    # IndexSearcher(ir);


# it assumes that run files are sorted
# currently it is only taking retrieved paras for top level section queries
def convert_para_run_to_topsec_para(para_run, top_k):
    run_as_dict = dict()
    topsec_para = dict()
    with open(para_run,'r') as pr:
        for l in pr:
            q = l.split(" ")[0]
            p = l.split(" ")[2]
            if q.count('/') < 2:
                if "/" in q:
                    page = q.split("/")[0]
                    topsec = q.split("/")[1]
                else:
                    page = q
                    topsec = q
                if page not in run_as_dict.keys():
                    run_as_dict[page] = dict()
                    run_as_dict[page][topsec] = [p]
                else:
                    if topsec not in run_as_dict[page].keys():
                        run_as_dict[page][topsec] = [p]
                    else:
                        run_as_dict[page][topsec].append(p)
    for page in run_as_dict.keys():
        topsec_para[page] = dict()
        for sec in run_as_dict[page].keys():
            topsec_para[page][sec] = []
            for p in run_as_dict[page][sec]:
                topsec_para[page][sec].append(p)
                if len(topsec_para[page][sec]) == top_k:
                    break
    return topsec_para

def get_set_of_paras(page_sec_para_dict, page_paras_json, output_file):
    paraset = set()
    for page in page_sec_para_dict.keys():
        for sec in page_sec_para_dict[page].keys():
            paraset = paraset.union(set(page_sec_para_dict[page][sec]))
    with open(page_paras_json, 'r') as art:
        page_paras = json.load(art).values()
    page_paras_list = []
    for p in page_paras:
        page_paras_list.extend(p)
    paraset = paraset.union(set(page_paras_list))
    with open(output_file, 'w') as out:
        for p in paraset:
            out.write(p+"\n")

def get_para_text_lucene(paraid, searcher):
    #searcher = IndexSearcher(DirectoryReader.open(SimpleFSDirectory(Paths.get(dir_path))))
    qp = QueryParser("Id", StandardAnalyzer())
    q = qp.parse(paraid)
    para_text = searcher.doc(searcher.search(q, 1).scoreDocs[0].doc).get("Text")
    print(para_text)

def preprocess_paras(paratext_json, stemlem):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    with open(paratext_json, 'r') as fp:
        paratext_dict = json.load(fp)
    preprocessed_paratext = dict()
    count = 1
    for paraid in paratext_dict.keys():
        paratext = str(paratext_dict[paraid])
        pt_low = paratext.lower()
        pt_nonum = re.sub(r'\d+', '', pt_low)
        pt_strip = pt_nonum.strip()
        tokens = word_tokenize(pt_strip)
        pun_table = str.maketrans('', '', string.punctuation)
        tokens_nopun = [t.translate(pun_table) for t in tokens]
        tokens_nostop = [i for i in tokens_nopun if not i in stop_words]
        if stemlem == "s":
            tokens_stemlem = [stemmer.stem(i) for i in tokens_nostop]
        elif stemlem == "l":
            tokens_stemlem = [lemmatizer.lemmatize(i) for i in tokens_nostop]
        else:
            tokens_stemlem = tokens_nostop
        tokens_final = [i for i in tokens_stemlem if i != '']
        preprocessed_paratext[paraid] = tokens_final
        if count % 100 == 0:
            print(count)
        count += 1
    return preprocessed_paratext

def get_para_token_freq(preproc_paras):
    para_token_freq = dict()
    for para in preproc_paras[()].keys():
        para_tokens = preproc_paras[()][para]
        token_freqs = dict()
        for t in para_tokens:
            token_freqs[t] = para_tokens.count(t)
        para_token_freq[para] = token_freqs
    return para_token_freq

def save_para_token_freq(preproc_paras_np_file, output_file):
    preproc_paras = np.load(preproc_paras_np_file)
    para_token_freq = dict()
    for para in preproc_paras[()].keys():
        para_tokens = preproc_paras[()][para]
        token_freqs = dict()
        for t in para_tokens:
            token_freqs[t] = para_tokens.count(t)
        para_token_freq[para] = token_freqs
    np.save(output_file, np.array(para_token_freq))

def get_pagewise_seclm_variance(page_lm):
    for page in page_lm[()].keys():
        sec_lms_in_page = page_lm[()][page][2]
        mean_lm = np.mean([sec_lms_in_page[sec] for sec in sec_lms_in_page.keys()])
        js_variance = 0
        kl_variance = 0
        for sec in sec_lms_in_page.keys():
            js_variance += distance.jensenshannon(mean_lm, sec_lms_in_page[sec]) ** 2
            kl_variance += np.sum(special.kl_div(mean_lm, sec_lms_in_page[sec])) ** 2
        # print(page + ": " + str(js_variance / len(sec_lms_in_page.keys())))
        print(page + ": " + str(kl_variance / len(sec_lms_in_page.keys())))

def compress_triples_file(page_paras, triples_file):
    triples_dat = dict()
    with open(triples_file, 'r') as tr:
        for l in tr:
            page = l.split(" ")[0]
            p1 = int(l.split(" ")[1])
            p2 = int(l.split(" ")[2])
            p3 = int(l.split(" ")[3])
            if page not in triples_dat.keys():
                triples_dat[page] = []
                triples_dat[page].append([page_paras[page][p1], page_paras[page][p2], page_paras[page][p3]])
            else:
                triples_dat[page].append([page_paras[page][p1], page_paras[page][p2], page_paras[page][p3]])
    triples_dat = np.array(triples_dat)
    return triples_dat

def main():
    print("blah\n")
    # topsec_para = convert_para_run_to_topsec_para("/home/sumanta/Documents/Dugtrio-data/Odd-One-Out/by1train_candidate_para_runs/ghetto_sdm_paragraph.run", 100)
    # np.save("/home/sumanta/Documents/Dugtrio-data/Odd-One-Out/by1train_candidate_para_runs/ghetto_sdm_top100_paragraph_page_sec", np.array(topsec_para))
    # get_para_token_freq("/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup-preprocessed-para-token-dict/by1train_cand_top100+page_paras_preproc.npy",
    #                     "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup-preprocessed-para-token-dict/by1train_cand_top100+page_paras_preproc_tf.npy")
    # get_para_token_freq(
    #     "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup-preprocessed-para-token-dict/by1train_cand_top100+page_paras_preproc_lem.npy",
    #     "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup-preprocessed-para-token-dict/by1train_cand_top100+page_paras_preproc_lem_tf.npy")
    # get_para_token_freq(
    #     "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup-preprocessed-para-token-dict/by1train_cand_top100+page_paras_preproc_stem.npy",
    #     "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup-preprocessed-para-token-dict/by1train_cand_top100+page_paras_preproc_stem_tf.npy")

    preprocessed_para_tokens = preprocess_paras("/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1test-nodup.json.data/by1-test-nodup.para.texts.json", "")
    preprocessed_para_tokens_np = np.array(preprocessed_para_tokens)
    np.save("/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1test-nodup-preprocessed-para-token-dict/by1test_paras_preproc", preprocessed_para_tokens_np)

    #preprocessed_para_tokens = preprocess_paras("/home/sumanta/Documents/Dugtrio-data/Odd-One-Out/by1train_candidate_para_runs/ghetto_sdm_top100_paragraph_set.json", "")
    #preprocessed_para_tokens_np = np.array(preprocessed_para_tokens)
    #np.save("/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup-preprocessed-para-token-dict/by1train_cand_top100+page_paras_preproc", preprocessed_para_tokens_np)
    #page_lms = np.load("/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup-preprocessed-lm-data/by1train-nodup-stem-lmds-data.npy")
    #get_pagewise_seclm_variance(page_lms)

if __name__ == '__main__':
    main()