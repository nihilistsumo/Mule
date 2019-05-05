#!/usr/bin/python3

import para_preprocessor, math, json, os, sys
import numpy as np
from scipy.spatial import distance
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn import preprocessing
import spacy
from spacy.lang.en import English
from spacy import displacy
import logging

def preprocess_text(paratext):
    text = paratext.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ')  # get rid of problem chars
    text = ' '.join(text.split())
    return text

def get_elmo_embed_paras_in_page(page, page_paras, para_text_dict, page_para_labels, nlp, embed):
    paraids = []
    para_sentence_count = []
    for p in page_paras[page]:
        paraids.append(p)
    para_sentences = []
    para_sentences_sec_label = []
    para_sec_label = []
    for para in paraids:
        sec_label = page_para_labels[page][para]
        para_sec_label.append(sec_label)
        paratext = str(para_text_dict[para])
        text = preprocess_text(paratext)
        doc = nlp(text)
        sent_count = 0
        for i in doc.sents:
            if len(i) > 1:
                para_sentences.append(i.string.strip())
                para_sentences_sec_label.append(sec_label)
                sent_count += 1
        para_sentence_count.append(sent_count)
    embeddings = embed(para_sentences, signature="default", as_dict=True)["default"]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sentence_vecs = sess.run(embeddings)
    return sentence_vecs, paraids, para_sentence_count, para_sentences, para_sentences_sec_label, para_sec_label

def get_embeddings(page_paras, para_text_dict, page_para_labels):
    logging.getLogger('tensorflow').disabled = True
    embed_data = dict()
    nlp = spacy.load('en_core_web_md')
    url = "https://tfhub.dev/google/elmo/2"
    embed = hub.Module(url)
    for page in page_paras.keys():
        print(page)
        sentence_vecs, paraids, para_sentence_count, para_sentences, para_sentences_sec_label, para_sec_label = get_elmo_embed_paras_in_page(page, page_paras, para_text_dict, page_para_labels, nlp, embed)
        page_data = dict()
        page_data['sent_vecs'] = sentence_vecs
        page_data['paraids'] = paraids
        page_data['para_sent_count'] = para_sentence_count
        page_data['para_sentences'] = para_sentences
        page_data['para_sent_label'] = para_sentences_sec_label
        page_data['para_label'] = para_sec_label
        embed_data[page] = page_data
    return embed_data

def main():
    # by1train_nodup_json = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.para.texts.json"
    # page_paras_json = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.page.paras.json"
    # page_para_labels_json = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.page.para.labels.json"
    by1train_nodup_json = sys.argv[1]
    page_paras_json = sys.argv[2]
    page_para_labels_json = sys.argv[3]
    elmo_out_file = sys.argv[4]

    with open(by1train_nodup_json, 'r') as by:
        by1train_para_text_dict = json.load(by)
    with open(page_paras_json, 'r') as pp:
        by1train_page_paras = json.load(pp)
    with open(page_para_labels_json, 'r') as pl:
        by1train_labels = json.load(pl)

    embeddings_data = np.array(get_embeddings(by1train_page_paras, by1train_para_text_dict, by1train_labels))
    # np.save("/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup-elmo-data/by1train_elmo_data", embeddings_data)
    np.save(elmo_out_file, embeddings_data)
    print("Done")

if __name__ == '__main__':
    main()