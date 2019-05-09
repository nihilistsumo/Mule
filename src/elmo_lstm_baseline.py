#!/usr/bin/python3

import math, json, os, sys, random
import numpy as np
from scipy.spatial import distance
import tensorflow as tf
import tensorflow_hub as hub

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM

MAX_PARA_LEN = 10
BATCH_SIZE = 32
TRAINING_SPLIT = 0.8

def preprocess_text(paratext):
    text = paratext.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ')  # get rid of problem chars
    if len(text.split(" ")) >= MAX_PARA_LEN:
        text = ' '.join(text.split(" ")[:MAX_PARA_LEN])
    else:
        text = ' '.join(text.split(" "))
    return text

def load_parapair_data(parapair_dataset_file):
    data = np.load(parapair_dataset_file)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    paras_in_data = set()
    positive_samples = []
    negative_samples = []
    for k in data[()].keys():
        if data[()][k] == 1:
            positive_samples.append(k)
        else:
            negative_samples.append(k)
    for i in range(int(len(positive_samples) * TRAINING_SPLIT)):
        train_x.append(positive_samples[i])
        paras_in_data.add(positive_samples[i][0])
        paras_in_data.add(positive_samples[i][1])
    for i in range(int(len(negative_samples) * TRAINING_SPLIT)):
        train_x.append(negative_samples[i])
        paras_in_data.add(negative_samples[i][0])
        paras_in_data.add(negative_samples[i][1])
    random.shuffle(train_x)
    for x in train_x:
        train_y.append(data[()][x])

    for i in range(int(len(positive_samples) * TRAINING_SPLIT), len(positive_samples)):
        test_x.append(positive_samples[i])
        paras_in_data.add(positive_samples[i][0])
        paras_in_data.add(positive_samples[i][1])
    for i in range(int(len(negative_samples) * TRAINING_SPLIT), len(positive_samples)):
        test_x.append(negative_samples[i])
        paras_in_data.add(negative_samples[i][0])
        paras_in_data.add(negative_samples[i][1])
    random.shuffle(test_x)
    for x in test_x:
        test_y.append(data[()][x])

    return train_x, train_y, test_x, test_y, list(paras_in_data)

def get_para_elmo_embeddings(paras_in_data, para_texts_json):
    url = "https://tfhub.dev/google/elmo/2"
    embed = hub.Module(url, trainable=True)
    with open(para_texts_json, 'r') as pt:
        para_text_dict = json.load(pt)
    para_texts = []
    print("Preprocessing paras...")
    for para in paras_in_data:
        print(para)
        text = str(para_text_dict[para])
        text = preprocess_text(text)
        para_texts.append(text)
    embeddings_dict = embed(para_texts, signature="default", as_dict=True)
    ######################################################
    #### This code is to test concat of elmo vectors
    #
    # wemb = embeddings_dict["word_emb"]
    # lstm1 = embeddings_dict["lstm_outputs1"]
    # lstm2 = embeddings_dict["lstm_outputs2"]
    # wt_mean_emb = embeddings_dict["elmo"]
    # def_emb = embeddings_dict["default"]
    # embeddings_para_tokens = tf.concat([wemb, lstm1, lstm2], axis=2)
    # embeddings_para = tf.reduce_mean(embeddings_para_tokens, axis=1)
    ######################################################
    print("Obtaining para elmo vectors...")
    embeddings_para = embeddings_dict["default"]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        para_vecs = sess.run(embeddings_para)
    print("Done")
    return para_vecs

def main():
    # parapair_dataset_file = "/home/sumanta/Documents/Dugtrio-data/Parapair_LSTM_data/by1train-tiny-parapair.npy"
    # para_text_json = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.para.texts.json"
    parapair_dataset_file = sys.argv[1]
    para_text_json = sys.argv[2]
    outfile = sys.argv[3]
    train_x, train_y, test_x, test_y, paras = load_parapair_data(parapair_dataset_file)
    para_elmo_vecs = get_para_elmo_embeddings(paras, para_text_json)
    para_elmo_vecs_dict = dict()
    for i in range(len(paras)):
        para = paras[i]
        para_elmo_vecs_dict[para] = para_elmo_vecs[i]
    np.save(outfile, para_elmo_vecs_dict)

if __name__ == '__main__':
    main()