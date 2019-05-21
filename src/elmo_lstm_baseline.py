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

TRAINING_SPLIT = 0.8

def load_parapair_data(parapair_dataset_file):
    data = np.load(parapair_dataset_file, allow_pickle=True)
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

def main():
    # parapair_dataset_file = "/home/sumanta/Documents/Dugtrio-data/Parapair_LSTM_data/by1train-tiny-parapair.npy"
    # para_text_json = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.para.texts.json"
    parapair_dataset_file = sys.argv[1]
    para_text_json = sys.argv[2]
    para_elmo_vecs_file = sys.argv[3]
    outfile = sys.argv[4]
    train_x, train_y, test_x, test_y, paras = load_parapair_data(parapair_dataset_file)
    para_elmo_vecs_dict = np.load(para_elmo_vecs_file)

if __name__ == '__main__':
    main()