#!/usr/bin/python3

import numpy as np
import tensorflow as tf
import json, os, argparse
from scipy import stats

def rev(parapair):
    p1 = parapair.split("_")[0]
    p2 = parapair.split("_")[1]
    return p2+"_"+p1

def load_parapairs(parapair_file):
    with open(parapair_file, 'r') as ppf:
        parapairs = json.load(ppf)
    return parapairs

def load_parapair_scores(score_filedir):
    score_dict = dict()
    for fname in os.listdir(score_filedir):
        with open(score_filedir+"/"+fname, 'r') as sf:
            score_dict[fname] = json.load(sf)
    return score_dict

def prepare_train_data(score_dict, parapairs_dict):
    feature_list = list(score_dict.keys())
    Xtrain = []
    ytrain = []
    for page in parapairs_dict.keys():
        parapairs = parapairs_dict[page]["parapairs"]
        for i in range(len(parapairs)):
            pp = parapairs[i]
            fet_scores = []
            for j in range(len(feature_list)):
                fet = feature_list[j]
                if pp in score_dict[fet].keys():
                    fet_scores.append(score_dict[fet][pp])
                else:
                    fet_scores.append(score_dict[fet][rev(pp)])
            Xtrain.append(fet_scores)
            ytrain.append(parapairs_dict[page]["labels"][i])
        print(page)
    Xtrain = np.array(Xtrain)
    print("Xtrain shape: "+str(Xtrain.shape))
    ytrain = np.array(ytrain).reshape(len(ytrain), 1)
    print("ytrain shape: "+str(ytrain.shape))
    return Xtrain, ytrain, feature_list

def train_model(Xtrain, ytrain, fet_list):
    w = tf.Variable(np.zeros((len(fet_list), 1)), trainable=True, dtype=tf.float64)
    x = tf.convert_to_tensor(Xtrain)
    y = tf.convert_to_tensor(ytrain)
    y_pred = tf.matmul(x, w)
    mse = tf.losses.mean_squared_error(y, y_pred)
    cost = tf.reduce_mean(mse)
    adam = tf.train.AdamOptimizer(learning_rate=0.01)
    a = adam.minimize(mse, var_list=w)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            sess.run(a)
            if i % 1000 == 0:
                print(str(sess.run(cost)))
        optimized_fet_weight = sess.run(w).reshape((w.shape[0], 1))
        print("Estimated weight: " + str(optimized_fet_weight))

    return optimized_fet_weight

def main():
    parser = argparse.ArgumentParser(description="Train a model to combine parapair scores")
    parser.add_argument("-pp", "--parapair_file", required=True, help="Path to train parapair file")
    parser.add_argument("-pps", "--parapair_score_dir", required=True, help="Path to directory containing parapair scores to be combined")
    parser.add_argument("-o", "--model_out", required=True, help="Path to output")
    args = vars(parser.parse_args())
    parapair_file = args["parapair_file"]
    parapair_score_dir = args["parapair_score_dir"]
    outpath = args["model_out"]

    parapairs_dict = load_parapairs(parapair_file)
    parapair_scores = load_parapair_scores(parapair_score_dir)
    Xtrain, ytrain, fet_list = prepare_train_data(parapair_scores, parapairs_dict)
    Xtrain = stats.zscore(Xtrain, axis=0)
    print(Xtrain[10:])
    opt_weights = train_model(Xtrain, ytrain, fet_list)
    model_dict = dict()

    for f in range(len(fet_list)):
        model_dict[fet_list[f]] = opt_weights[f][0]

    with open(outpath, 'w') as out:
        json.dump(model_dict, out)

if __name__ == '__main__':
    main()