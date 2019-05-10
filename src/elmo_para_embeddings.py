#!/usr/bin/python3

import math, json, os, sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import spacy
import logging

def preprocess_text(paratext):
    text = paratext.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ')  # get rid of problem chars
    text = ' '.join(text.split())
    return text

def get_elmo_embed_paras(paras, para_text_dict, nlp, embed):
    print(len(paras)+" total paras")
    paraids = []
    para_sentences = []
    for para in paras:
        paratext = str(para_text_dict[para])
        text = preprocess_text(paratext)
        doc = nlp(text)
        for i in doc.sents:
            if len(i) > 1:
                para_sentences.append(i.string.strip())
                paraids.append(para)
    print(len(para_sentences) + " total sentences")
    embed_dict = embed(para_sentences, signature="default", as_dict=True)
    wemb = embed_dict["word_emb"]
    lstm1 = embed_dict["lstm_outputs1"]
    lstm2 = embed_dict["lstm_outputs2"]
    embeddings = tf.concat([wemb, lstm1, lstm2], axis=2)

    print("Starting tensorflow session...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        embed_vecs = sess.run(embeddings)
    return embed_vecs, paraids

def main():
    para_list = sys.argv[1]
    para_text_file = sys.argv[2]
    outfile = sys.argv[3]
    with open(para_list, 'r') as pl:
        paras = json.load(pl)
    with open(para_text_file, 'r') as pt:
        para_text_dict = json.load(pt)

    logging.getLogger('tensorflow').disabled = True
    embed_data = dict()
    nlp = spacy.load('en_core_web_md')
    url = "https://tfhub.dev/google/elmo/2"
    embed = hub.Module(url)
    embed_vecs, paraids = get_elmo_embed_paras(paras, para_text_dict, nlp, embed)
    embed_data['paras'] = paraids
    embed_data['vecs'] = embed_vecs
    print("Done")
    np.save(outfile, embed_vecs)

if __name__ == '__main__':
    main()