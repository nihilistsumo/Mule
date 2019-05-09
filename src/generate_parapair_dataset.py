#!/usr/bin/python3

import random, json, os, sys
import numpy as np

def generate_data(simcount, oddcount_samepage, oddcount_diffpage, page_para_labels):
    tiny_by1train_parapair_data = dict()
    count = 0
    while (count < simcount):
        page = random.sample(page_para_labels.keys(), 1)[0]
        para = random.sample(page_para_labels[page].keys(), 1)[0]
        paralabel = page_para_labels[page][para]
        for p in page_para_labels[page].keys():
            if para != p and paralabel == page_para_labels[page][p]:
                simpara = p
                if (para, simpara) not in tiny_by1train_parapair_data.keys() and (
                simpara, para) not in tiny_by1train_parapair_data.keys():
                    tiny_by1train_parapair_data[(para, simpara)] = 1
                    count += 1
                    break

    count = 0
    while (count < oddcount_samepage):
        page = random.sample(page_para_labels.keys(), 1)[0]
        para = random.sample(page_para_labels[page].keys(), 1)[0]
        paralabel = page_para_labels[page][para]
        for p in page_para_labels[page].keys():
            if para != p and paralabel != page_para_labels[page][p]:
                oddpara = p
                if (para, oddpara) not in tiny_by1train_parapair_data.keys() and (oddpara, para) not in tiny_by1train_parapair_data.keys():
                    tiny_by1train_parapair_data[(para, oddpara)] = 0
                    count += 1
                    break

    count = 0
    while (count < oddcount_diffpage):
        page = random.sample(page_para_labels.keys(), 1)[0]
        para = random.sample(page_para_labels[page].keys(), 1)[0]
        otherpage = random.sample(page_para_labels.keys(), 1)[0]
        while (page == otherpage):
            otherpage = random.sample(page_para_labels.keys(), 1)[0]
        oddpara = random.sample(page_para_labels[otherpage].keys(), 1)[0]
        if (para, oddpara) not in tiny_by1train_parapair_data.keys() and (oddpara, para) not in tiny_by1train_parapair_data.keys():
            tiny_by1train_parapair_data[(para, oddpara)] = 0
            count += 1

    return tiny_by1train_parapair_data

def main():
    page_para_labels_json = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.page.para.labels.json"
    output_file = "/home/sumanta/Documents/Dugtrio-data/Parapair_LSTM_data/by1train-tiny-parapair"
    with open(page_para_labels_json, 'r') as pl:
        page_para_labels = json.load(pl)
    dataset = generate_data(500, 500, 500, page_para_labels)
    np.save(output_file, dataset)

if __name__ == '__main__':
    main()