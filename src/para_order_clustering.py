#!/usr/bin/python3

import json, re, string, os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
import numpy as np

def get_para_text_dict(page_json_data, stemlem):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    para_text_dict = dict()
    for p in page_json_data['paragraphs']:
        paraid = p["para_id"]
        paratext = ""
        for pt in p["para_body"]:
            paratext += pt["text"] + " "
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
        para_text_dict[paraid] = tokens_final
    return para_text_dict

def get_idf(page_para_token_dict):
    df_dict = dict()
    doc_num = 0
    idf_dict = dict()
    for page in page_para_token_dict.keys():
        for para in page_para_token_dict[page].keys():
            doc_num += 1
            tokenized_para = page_para_token_dict[page][para]
            for t in set(tokenized_para):
                if t in df_dict.keys():
                    df_dict[t] += 1
                else:
                    df_dict[t] = 1
    for t in df_dict.keys():
        idf_dict[t] = np.log(doc_num) - np.log(1 + df_dict[t])
    return idf_dict, df_dict

def get_tfidf(tokenized_para, idf_dict, all_terms):
    tfidf_vec = np.zeros(len(all_terms))
    tf_vec = np.zeros(len(all_terms))
    for i in range(len(all_terms)):
        t = all_terms[i]
        if t in tokenized_para:
            tf_vec[i] = tokenized_para.count(t)
    for i in range(len(all_terms)):
        t = all_terms[i]
        tfidf_vec[i] = tf_vec[i] * idf_dict[t]
    return tfidf_vec

def get_rand(true_labels, obtained_labels):
    a = 0
    b = 0
    n = len(true_labels)
    for i in range(len(true_labels)-1):
        for j in range(i+1, len(true_labels)):
            if len(set(true_labels[i]).intersection(set(true_labels[j]))) > 0 and obtained_labels[i] == obtained_labels[j]:
                a += 1
            elif len(set(true_labels[i]).intersection(set(true_labels[j]))) == 0 and obtained_labels[i] != obtained_labels[j]:
                b += 1
    return 2*(a + b)/(n * (n-1))

def get_para_labels_from_qrels(qrels_file):
    para_label_dict = dict()
    with open(qrels_file, 'r') as q:
        for l in q:
            para = l.split(" ")[2]
            sec = l.split(" ")[0]
            if para not in para_label_dict.keys():
                para_label_dict[para] = [sec]
            else:
                para_label_dict[para].append(sec)
    return para_label_dict

def convert_json_tfidf_dict(ordering_json_dir, stemlem):
    page_para_token_dict = dict()
    for f in os.listdir(ordering_json_dir):
        with open(ordering_json_dir+"/"+f, 'r') as ord:
            ordering_json_data = json.load(ord)
        page_para_token_dict[f] = get_para_text_dict(ordering_json_data, stemlem)
    idf_dict, df_dict = get_idf(page_para_token_dict)
    all_terms = list(df_dict.keys())
    page_para_tfidf = dict()
    for page in page_para_token_dict.keys():
        page_para_tfidf[page] = dict()
        for para in page_para_token_dict[page]:
            page_para_tfidf[page][para] = get_tfidf(page_para_token_dict[page][para], idf_dict, all_terms)
    return page_para_tfidf

def cluster_paras_page(page, page_para_tfidf, true_labels_dict, output):
    paras_in_page = list(page_para_tfidf[page].keys())
    true_labels = []
    obtained_labels = []
    para_vecs = []
    for p in range(len(paras_in_page)):
        para = paras_in_page[p]
        if para not in true_labels_dict.keys():
            continue
        para_vecs.append(page_para_tfidf[page][para])
        true_labels.append(true_labels_dict[para])
    secs_in_page = set()
    for l in true_labels:
        secs_in_page = secs_in_page.union(l)
    true_k = len(secs_in_page)
    if true_k == 0:
        print(page+": No relevant para retrieved")
        output.write(page+": No relevant para retrieved\n")
    elif true_k == 1:
        print(page + ": Only relevant paras from a single true cluster got retrieved")
        output.write(page + ": Only relevant paras from a single true cluster got retrieved\n")
    elif true_k > len(para_vecs):
        print(page + ": No of relevant paragraphs retrieved are lower than no of unique sections (union of true labels)")
        output.write(page + ": No of relevant paragraphs retrieved are lower than no of unique sections (union of true labels)\n")
    else:
        kmeans = KMeans(n_clusters=true_k, random_state=0)
        obtained_labels = kmeans.fit_predict(para_vecs)
    return true_labels, obtained_labels

def main():
    json_data_dir = "/home/sumanta/Documents/jordan_Y2_para_orderings/json_data"
    qrels = "/home/sumanta/Documents/jordan_Y2_para_orderings/qrels/benchmarkY2test-psg-manual-toplevel.qrels"
    output_dir = "/home/sumanta/Documents/jordan_Y2_para_orderings/clustering_results"
    #for m in os.listdir(json_data_dir):
    m = "DWS-UMA-SemqQueryExp"
    print(m)
    print("===================================\n")
    page_para_tfidf = convert_json_tfidf_dict(json_data_dir+"/"+m, "s")
    true_labels_dict = get_para_labels_from_qrels(qrels)
    output_file = output_dir+"/"+m+".result"
    with open(output_file, 'w+') as out:
        for page in page_para_tfidf.keys():
            true_labels, obtained_labels = cluster_paras_page(page, page_para_tfidf, true_labels_dict, out)
            if len(true_labels) > 0 and len(obtained_labels) > 0:
                print(page+" RAND: "+str(get_rand(true_labels, obtained_labels)))
                out.write(page+" RAND: "+str(get_rand(true_labels, obtained_labels))+"\n")

if __name__ == '__main__':
    main()