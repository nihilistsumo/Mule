#!/udfsr/bin/python3

import para_preprocessor, math, json, statistics, sys, argparse
import numpy as np
from scipy.spatial import distance
import preproc_baseline

## This is to calculate parapair scores for bm25 and tfidf baselines
## To calculate parapair scores for elmo vecs, see elmo_baseline.py

parser = argparse.ArgumentParser(description="Calculate parapair scores and save them in a dict")
parser.add_argument("-pgp", "--page_paras", required=True, help="Path to page-paras json file")
parser.add_argument("-pp", "--parapairs", required=True, help="Path to parapairs file")
parser.add_argument("-t", "--preproc_token", required=True, help="Path to preprocessed token file")
parser.add_argument("-m", "--method", required=True, help="Method name (bm25/tfidf)")
parser.add_argument("-o", "--out", required=True, help="Path to output file")
args = vars(parser.parse_args())
page_paras_json_file = args["page_paras"]
parapairs_file = args["parapairs"]
preproc_para_tokens_file = args["preproc_token"]
method = args["method"]
outfile = args["out"]

with open(parapairs_file, 'r') as ppf:
    pp_data = json.load(ppf)
# parapairs = pp_data['parapairs']
preproc_para_token_freq = para_preprocessor.get_para_token_freq(np.load(preproc_para_tokens_file))
all_terms = preproc_baseline.get_terms_list(preproc_para_token_freq)
df = preproc_baseline.get_df(all_terms, preproc_para_token_freq)
avg_doclen, doc_lens = preproc_baseline.get_doclen_stats(preproc_para_token_freq)
with open(page_paras_json_file, 'r') as pf:
    page_paras = json.load(pf)
paras = []
for p in page_paras.keys():
    paras.extend(page_paras[p])
para_tfidf_dict = preproc_baseline.get_para_tfidf_vec_page(paras, preproc_para_token_freq, df)
para_tfidf = dict()
for p in para_tfidf_dict.keys():
    para_tfidf[p] = preproc_baseline.expand_term_vec(para_tfidf_dict[p], all_terms)

if method == 'bm25':
    print("bm25")
    parapair_scores = preproc_baseline.get_parapair_bm25_scores(pp_data, preproc_para_token_freq, df, avg_doclen, doc_lens)
else:
    print("tfidf")
    parapair_scores = preproc_baseline.get_parapair_tfidf_scores(pp_data, para_tfidf)

with open(outfile, 'w') as out:
    json.dump(parapair_scores, out)