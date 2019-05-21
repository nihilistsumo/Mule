#!/usr/bin/python3

import math, json, os, sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import spacy
import logging

#######################################
#
# Single swipe through para list
#
#######################################

def preprocess_text(paratext):
    text = paratext.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ')  # get rid of problem chars
    text = ' '.join(text.split())
    return text

def get_elmo_embed_paras(paras, para_text_dict, nlp, embed):
    print(str(len(paras))+" total paras")
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
    print(str(len(para_sentences)) + " total sentences")
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
    outdir = sys.argv[3]
    if len(sys.argv) > 4:
        tf_cache_dir_path = sys.argv[4]
        os.environ['TFHUB_CACHE_DIR'] = tf_cache_dir_path
    with open(para_list, 'r') as pl:
        paras = json.load(pl)
    with open(para_text_file, 'r') as pt:
        para_text_dict = json.load(pt)

    logging.getLogger('tensorflow').disabled = True
    nlp = spacy.load('en_core_web_md')
    url = "https://tfhub.dev/google/elmo/2"
    embed = hub.Module(url)

    embed_vecs, paraids = get_elmo_embed_paras(paras, para_text_dict, nlp, embed)
    print("Done")
    np.save(outdir+"/embeddings_vecs", embed_vecs)
    np.save(outdir+"/corresponding_paraids", paraids)

if __name__ == '__main__':
    main()