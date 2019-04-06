#!/usr/bin/python3
import sys
import numpy as np

def get_toplevel_topics(topics_file):
    topics = dict()
    with open(topics_file, 'r') as top:
        for l in top:
            if l != "\n":
                page = l.split("/")[0].rstrip()
                toplevel = l.split("/")[1].rstrip()
                if page not in topics.keys():
                    topics[page] = set([toplevel])
                else:
                    topics[page].add(toplevel)
    for k in topics.keys():
        topics[k] = list(topics[k])
    return np.array(topics)

def get_page_paras(art_qrels):
    page_paras = dict()
    with open(art_qrels, 'r') as art:
        for l in art:
            page = l.split(" ")[0]
            para = l.split(" ")[2]
            if page not in page_paras.keys():
                page_paras[page] = [para]
            else:
                page_paras[page].append(para)
    return np.array(page_paras)