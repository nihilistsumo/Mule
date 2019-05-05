#!/usr/bin/python3
import concurrent
import json, sys, math, time
import numpy as np
from scipy import spatial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def get_page_term_freq_vecs(page_paras_file, page_topics_file, labels_file, terms_file, para_ind_file, para_freq_file):
    with open(terms_file,'r') as tf:
        terms = json.load(tf)
    with open(labels_file,'r') as lf:
        labels = json.load(lf)
    with open(page_paras_file,'r') as pf:
        page_paras = json.load(pf)
    with open(page_topics_file,'r') as tf:
        page_tops = json.load(tf)
    with open(para_freq_file,'r') as ff:
        freqs = json.load(ff)
    with open(para_ind_file,'r') as indf:
        inds = json.load(indf)

    per_label_freq_vec = dict()
    for page in page_paras:
        print(page)
        labels_in_page = labels[page]
        label_term_freq_page = dict()
        for l in set(labels_in_page.values()):
            label_term_freq_page[l] = np.zeros(len(terms))

        for p in page_paras[page]:
            label = labels_in_page[p]
            for i in range(len(inds[p])):
                label_term_freq_page[label][inds[p][i]] += freqs[p][i]

        per_label_freq_vec[page] = label_term_freq_page

    return per_label_freq_vec, terms, labels, page_paras, page_tops, inds, freqs

def dirichlet_smoothing(vec, c_vec, mu=1):
    smoothed_vec = np.zeros(len(vec))
    d = np.sum(vec)
    for i in range(len(vec)):
        smoothed_vec[i] = (vec[i] + mu*c_vec[i])/(d + mu)
    return smoothed_vec

def jm_smoothing(vec, c_vec, lamb=0.9):
    smoothed_vec = np.zeros(len(vec))
    d = np.sum(vec)
    C = np.sum(c_vec)
    if d<1:
        smoothed_vec = c_vec / C
    else:
        for i in range(len(vec)):
            smoothed_vec[i] = lamb*vec[i]/d + (1-lamb)*c_vec[i]/C
    return smoothed_vec

def jm_smoothing_idf(vec, c_vec, lamb=0.9):
    return jm_smoothing(vec, c_vec, lamb) * idf_vec

def get_tfidf_vec(freq_vec, idf_vec):
    return (freq_vec/np.sum(freq_vec)) * idf_vec

def jsdiv_dist(p,q,p_log,q_log,big_number=10):
    if len(p) != len(q):
        raise Exception("Can not calculate JSdiv of two different length vectors: "+str(len(p))+","+str(len(q)))
    m = (p+q)/2
    m_log = np.log(m)
    for x in range(len(m_log)):
        if m_log[x] < sys.float_info.min:
            m_log[x] = -sys.float_info.max
    return math.sqrt((kldiv(p,m,p_log,m_log,big_number) + kldiv(q,m,q_log,m_log,big_number))/2)

def kldiv(p,q,p_log,q_log,big_number=10):
    # big_number is used to avoid calc of log(q[x]) when q[x] is very small
    if len(p) != len(q):
        raise Exception("Can not calculate KLdiv of two different length vectors: "+str(len(p))+","+str(len(q)))
    score = 0
    for x in range(len(p)):
        if p[x] < sys.float_info.min:
            continue # because lim x.log(x) = 0 when x -> 0+
        elif q[x] < sys.float_info.min:
            score += big_number
        else:
            comp = p[x] * (p_log[x] - q_log[x])
            score += comp
    return score

def kldiv_mt(p,q,p_log,q_log,i,data,big_number=10):
    # big_number is used to avoid calc of log(q[x]) when q[x] is very small
    data[i] = kldiv(p, q, p_log, q_log, big_number)

def simple_distribution_dist(p,q):
    if len(p) != len(q):
        raise Exception("Can not calculate distribution distance of two different length vectors: "+str(len(p))+","+str(len(q)))
    score = 0
    for x in range(len(p)):
        score += abs(p[x] - q[x])/((p[x] + q[x])/2)
    return score

correct = []
incorrect = []
def calc_para_lm(page, smoothing):
    cum_page_freq = np.zeros(len(terms))
    for k in page_freq_vec[page].keys():
        cum_page_freq += page_freq_vec[page][k]
    c_term_freq = np.array(global_term_freq - cum_page_freq)

    paras = page_paras[page]
    para_lm = dict()
    para_term_freq_vecs = dict()
    for p in paras:
        pind = para_inds[p]
        pfreq = para_freqs[p]
        p_term_vec = np.zeros(len(terms))
        for i in range(len(pind)):
            p_term_vec[pind[i]] = pfreq[i]
        para_term_freq_vecs[p] = p_term_vec
        if smoothing == 'jm':
            para_lm[p] = jm_smoothing(p_term_vec, c_term_freq)
        elif smoothing == 'ds':
            para_lm[p] = dirichlet_smoothing(p_term_vec, c_term_freq)
        else:
            para_lm[p] = p_term_vec
    return para_lm, para_term_freq_vecs, c_term_freq

def calc_para_tfidf(page):
    paras = page_paras[page]
    para_tfidf = dict()
    for p in paras:
        pind = para_inds[p]
        pfreq = para_freqs[p]
        p_term_vec = np.zeros(len(terms))
        for i in range(len(pind)):
            p_term_vec[pind[i]] = pfreq[i]
        para_tfidf[p] = get_tfidf_vec(p_term_vec, idf_vec)
    return para_tfidf

def calc_accuracy_tfidf(page):
    print(page + " started " + str(len(triples_data[()][page])) + " triples")
    para_tfidf = calc_para_tfidf(page)
    acc = 0.0
    for t in triples_data[()][page]:
        dist_p12 = spatial.distance.cosine(para_tfidf[t[0]], para_tfidf[t[1]])
        dist_p23 = spatial.distance.cosine(para_tfidf[t[1]], para_tfidf[t[2]])
        dist_p31 = spatial.distance.cosine(para_tfidf[t[2]], para_tfidf[t[0]])
        if (dist_p12 < dist_p23 and dist_p12 < dist_p31):
            acc += 1
            correct.append(True)
            #print("*")
        else:
            incorrect.append(True)
            #print("-")
    accuracy = acc * 100.0 / len(triples_data[()][page])
    print(page + " Accuracy: " + str(accuracy) + " %")

def calc_accuracy(page):
    if(len(triples_data[()][page]) < max_num_triples_in_page):
        print(page+" started "+str(len(triples_data[()][page]))+" triples")
        para_lmjm, para_term_freq_vecs, c_term_freq = calc_para_lmjm(page)
        acc = 0.0
        for t in triples_data[()][page]:
            top_freq_vec_page = page_freq_vec[page]
            p_term_vecs = []
            for p in t:
                plabel = page_para_labels[page][p]
                top_freq_vec_page[plabel] -= para_term_freq_vecs[p]
                p_term_vecs.append(para_lmjm[p])

            lm_in_page = []
            for l in top_freq_vec_page.keys():
                lm_in_page.append(jm_smoothing_idf(top_freq_vec_page[l], c_term_freq))
            lm_in_page = np.array(lm_in_page)

            kldiv_vecs = []
            for i in range(3):
                kldiv_vec = []
                for l in range(len(lm_in_page)):
                    kldiv_vec.append(kldiv(p_term_vecs[i], lm_in_page[l]))
                kldiv_vec = np.array(kldiv_vec)
                kldiv_vec = kldiv_vec / np.sum(kldiv_vec)
                kldiv_vecs.append(kldiv_vec)
                # print(str(i+1)+": "+t[i]+": "+str(kldiv_vec.tolist()))
            dist_p12 = simple_distribution_dist(kldiv_vecs[0], kldiv_vecs[1])
            dist_p23 = simple_distribution_dist(kldiv_vecs[1], kldiv_vecs[2])
            dist_p31 = simple_distribution_dist(kldiv_vecs[2], kldiv_vecs[0])
            if (dist_p12 < dist_p23 and dist_p12 < dist_p31):
                acc += 1
                correct.append(True)
                #print("*")
            else:
                incorrect.append(True)
                #print("-")
            # print("Similar paras: ("+t[0]+","+t[1]+"), Odd: "+t[2])
            # print("1,2: "+str(kldiv(kldiv_vecs[0],kldiv_vecs[1]))+", 2,3: "+str(kldiv(kldiv_vecs[1],kldiv_vecs[2]))+
            #       ", 3,1: "+str(kldiv(kldiv_vecs[2],kldiv_vecs[0])))
        accuracy = acc * 100.0 / len(triples_data[()][page])
        print(page + " Accuracy: " + str(accuracy) + " %")

def calc_accuracy_jsdiv(page):
    if(len(triples_data[()][page]) < max_num_triples_in_page):
        print(page+" started "+str(len(triples_data[()][page]))+" triples")
        para_lmjm, para_term_freq_vecs, c_term_freq = calc_para_lmjm(page)
        acc = 0.0
        for t in triples_data[()][page]:
            top_freq_vec_page = page_freq_vec[page]
            p_term_vecs = []
            for p in t:
                plabel = page_para_labels[page][p]
                top_freq_vec_page[plabel] -= para_term_freq_vecs[p]
                p_term_vecs.append(para_lmjm[p])

            lm_in_page = []
            for l in top_freq_vec_page.keys():
                lm_in_page.append(jm_smoothing_idf(top_freq_vec_page[l], c_term_freq))
            lm_in_page = np.array(lm_in_page)

            jsdiv_vecs = []
            for i in range(3):
                jsdiv_vec = []
                for l in range(len(lm_in_page)):
                    jsdiv_vec.append(kldiv(p_term_vecs[i], lm_in_page[l]))
                jsdiv_vec = np.array(jsdiv_vec)
                jsdiv_vec = jsdiv_vec / np.sum(jsdiv_vec)
                jsdiv_vecs.append(jsdiv_vec)
                # print(str(i+1)+": "+t[i]+": "+str(kldiv_vec.tolist()))
            dist_p12 = simple_distribution_dist(jsdiv_vecs[0], jsdiv_vecs[1])
            dist_p23 = simple_distribution_dist(jsdiv_vecs[1], jsdiv_vecs[2])
            dist_p31 = simple_distribution_dist(jsdiv_vecs[2], jsdiv_vecs[0])
            if (dist_p12 < dist_p23 and dist_p12 < dist_p31):
                acc += 1
                correct.append(True)
                #print("*")
            else:
                incorrect.append(True)
                #print("-")
            # print("Similar paras: ("+t[0]+","+t[1]+"), Odd: "+t[2])
            # print("1,2: "+str(kldiv(kldiv_vecs[0],kldiv_vecs[1]))+", 2,3: "+str(kldiv(kldiv_vecs[1],kldiv_vecs[2]))+
            #       ", 3,1: "+str(kldiv(kldiv_vecs[2],kldiv_vecs[0])))
        accuracy = acc * 100.0 / len(triples_data[()][page])
        print(page + " Accuracy: " + str(accuracy) + " %")

def calc_paralm_dist_triple(t, para_lm, para_logs):
    dist = [0.0, 0.0, 0.0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.submit(kldiv_mt, para_lm[t[0]], para_lm[t[1]], para_logs[t[0]], para_logs[t[1]], 0, dist)
        executor.submit(kldiv_mt, para_lm[t[1]], para_lm[t[2]], para_logs[t[1]], para_logs[t[2]], 1, dist)
        executor.submit(kldiv_mt, para_lm[t[2]], para_lm[t[0]], para_logs[t[2]], para_logs[t[0]], 2, dist)

    if (dist[0] < dist[1] and dist[0] < dist[2]):
        return 1
        #print("*")
    else:
        return 0
        #print("-")

def calc_accuracy_paralm(page, lm):
    triples_count = len(triples_data[()][page])
    print(page+" started "+str(triples_count)+" triples")
    para_lm, para_term_freq_vecs, c_term_freq = calc_para_lm(page, lm)
    paras = page_paras[page]
    para_logs = dict()
    for p in paras:
        logs_in_p = np.log(para_lm[p])
        for x in range(len(logs_in_p)):
            if logs_in_p[x] < sys.float_info.min:
                logs_in_p[x] = 0
        para_logs[p] = logs_in_p
    acc = 0.0
    count = 0.0
    for t in triples_data[()][page]:
        count += 1
        acc += calc_paralm_dist_triple(t, para_lm, para_logs)
        print(str(round((count * 100 / triples_count), 2)) + " % completion: Accuracy " + str(
            round((acc * 100 / count), 4)) + " %", end='\r')
    accuracy = acc*100/count
    print("\n"+page + " Accuracy: " + str(accuracy) + " %")

def calc_accuracy_paralm_mt(page):
    triples_count = len(triples_data[()][page])
    print(page+" started "+str(triples_count)+" triples")
    para_lmjm, para_term_freq_vecs, c_term_freq = calc_para_lm(page)
    paras = page_paras[page]
    para_logs = dict()
    for p in paras:
        logs_in_p = []
        for x in range(len(para_lmjm[p])):
            if para_lmjm[p][x] < sys.float_info.min:
                logs_in_p.append(-sys.float_info.max)
            else:
                try:
                    logs_in_p.append(np.log(para_lmjm[p][x]))
                except Exception as e:
                    print("Exception generated in " + p)
        para_logs[p] = logs_in_p
    acc = 0.0
    count = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_triple = {executor.submit(calc_paralm_dist_triple, t, para_lmjm, para_logs): t for t in triples_data[()][page]}
        for future in concurrent.futures.as_completed(future_to_triple):
            triple = future_to_triple[future]
            count += 1
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (triple, exc))
            else:
                acc += data
                print(str(round((count*100/triples_count),2))+" % completion: Accuracy "+str(round((acc*100/count), 4))+" %", end='\r')
    accuracy = acc*100/count
    print("\n"+page + " Accuracy: " + str(accuracy) + " %\n")

def check_progress(num):
    total = 0
    while num != total:
        total = len(incorrect)+len(correct)
        print(total*100.0/num+"% triples evaluated, Mean accuracy so far: "+len(correct)*100.0/total)
        time.sleep(30)


terms_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.global.terms.json"
terms_freq_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.global.terms.freq.json"
terms_doc_freq_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.global.terms.doc.freq.json"
labels_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.page.para.labels.json"
page_paras_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.page.paras.json"
page_topics_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.page.tops.json"
para_freq_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.para.freqs.json"
para_ind_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.para.indices.json"
triples_file = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1-train-nodup.triples.npy"
max_num_triples_in_page = 500

# terms_file = sys.argv[1]
# terms_freq_file = sys.argv[2]
# terms_doc_freq_file = sys.argv[3]
# labels_file = sys.argv[4]
# page_paras_file = sys.argv[5]
# page_topics_file = sys.argv[6]
# para_freq_file = sys.argv[7]
# para_ind_file = sys.argv[8]
# triples_file = sys.argv[9]
# max_num_triples_in_page = int(sys.argv[10])

page_freq_vec, terms, page_para_labels, page_paras, page_tops, para_inds, para_freqs = \
    get_page_term_freq_vecs(page_paras_file, page_topics_file, labels_file, terms_file, para_ind_file, para_freq_file)

# triples_data = dict()
# with open(triples_file,'r') as tf:
#     for l in tf:
#         page = l.split(" ")[0]
#         #top_freq_vec_page = page_freq_vec[page]
#         t_paras = [page_paras[page][int(l.split(" ")[1])], page_paras[page][int(l.split(" ")[2])], page_paras[page][int(l.split(" ")[3])]]
#         if page in triples_data.keys():
#             triples_data[page].append(t_paras)
#         else:
#             triples_data[page] = [t_paras]
# np.save("/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1-train-nodup.triples",triples_data)

triples_data = np.load(triples_file)

with open(terms_freq_file,'r') as gtf:
    global_term_freq = json.load(gtf)
with open(terms_doc_freq_file,'r') as gdf:
    global_term_doc_freq = json.load(gdf)
doc_num = len(para_inds)
idf_vec = np.log(doc_num) - np.log(np.array(global_term_doc_freq))
print("Data loaded")

# pages_a has very few paras/triples
pages_a = ["enwiki:Chocolate%20chip",
"enwiki:Gaffkaemia",
"enwiki:Antibiotic%20misuse",
"enwiki:Biodiversity%20and%20food",
"enwiki:Egg%20white",
"enwiki:Contingent%20work",
"enwiki:Natural%20growth%20promoter",
"enwiki:Amphiprioninae",
"enwiki:Atmospheric%20sciences",
"enwiki:Water%20resource%20management"]

# pages_b has medium no. of paras/triples
pages_b = ["enwiki:Legionella",
"enwiki:Chocolate",
"enwiki:Agriprocessors",
"enwiki:Thermal%20runaway",
"enwiki:Fever",
"enwiki:Norepinephrine",
"enwiki:Informal%20sector",
"enwiki:Sweatshop",
"enwiki:Pollution",
"enwiki:Freshwater%20environmental%20quality%20parameters",
"enwiki:Rainbow%20trout",
"enwiki:Inflammatory%20bowel%20disease",
"enwiki:Sugar",
"enwiki:Environmental%20Justice%20Foundation",
"enwiki:Overfishing",
"enwiki:Credit%20rationing",
"enwiki:Blueberry",
"enwiki:Ion",
"enwiki:Soil%20erosion",
"enwiki:Coffeehouse"]

# pages_c either have very high no. of paras/triples or good distribution of paras across topics
pages_c = [
    "enwiki:Subprime%20mortgage%20crisis",
    "enwiki:Coffee",
    "enwiki:Human%20rights",
    "enwiki:Research%20in%20lithium-ion%20batteries",
    "enwiki:Oxygen",
    "enwiki:Photosynthesis",
    "enwiki:Natural%20resource%20management",
    "enwiki:Informal%20sector",
    "enwiki:Mole%20sauce"]

#for page in triples_data[()].keys():
start = time.time()
#for page in pages_with_less_paras:
for page in pages_b:
    #print(page)
    calc_accuracy_paralm(page, 'jm')
    #calc_accuracy_tfidf(page)
end = time.time()
print("Execution time: "+str(end-start)+" seconds")

# triples_count = 0
# for k in triples_data[()].keys():
#     triples_count += len(triples_data[()][k])

# with ProcessPoolExecutor(max_workers=num_workers) as ex:
#     for page in triples_data[()].keys():
#         ex.submit(calc_accuracy, page)

print("Finished")