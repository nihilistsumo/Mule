#!/usr/bin/python3

import math, json, os, sys
import numpy as np

def split_pagelist(pagelist, split_no):
    splits = []
    per_split_count = math.ceil(len(pagelist)/split_no)
    for i in range(split_no):
        split = []
        i = 0
        while(i < per_split_count and len(pagelist) > 0):
            split.append(pagelist[0])
            pagelist.pop(0)
            i += 1
        splits.append(split)
    return splits

def write_outputs(splits, output_dir):
    count = 1
    for s in splits:
        with open(output_dir+"/pages"+str(count), 'w') as p:
            for page in s:
                p.write(page+"\n")
        count += 1

def main():
    page_paras_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1test-nodup.json.data/by1-test-nodup.page.paras.json"
    outdir = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1test-nodup-page-parts"
    with open(page_paras_file, 'r') as pp:
        page_paras = json.load(pp)
    splits = split_pagelist(list(page_paras.keys()), 10)
    write_outputs(splits, outdir)

if __name__ == '__main__':
    main()