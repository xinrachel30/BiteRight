import os 
import pickle
import numpy as np
import math

vocab = []

BASE_DIR = os.getcwd()  # Get current working directory
DATA_DIR = os.path.join(BASE_DIR, "python_scripts_and_data", "data")

with open(os.path.join(DATA_DIR, "foodVocab.pkl"), "rb") as file: 
    vocab = pickle.load(file)

vocab = list(set(vocab))
vocab_dict = {}
i = 0
for food in vocab:   
    vocab_dict[food] = i
    i += 1

with open(os.path.join(DATA_DIR, "foodwords_score_dict.pkl"), "rb") as file:
    foodPair_upvotes = pickle.load(file)

def cosine_make_doc_term(foodPair_upvotes):
    doc_term = np.zeros((len(foodPair_upvotes),len(vocab)))
    # term doc matrix with tf = upvotes
    docIdx = 0
    for foodPair, uv in foodPair_upvotes.items():
        food_arr = [f.lower().strip() for f in foodPair.split(",") if f.strip()]
        for food in food_arr:
            doc_term[docIdx][vocab_dict[food]] = uv
        docIdx+=1
    return doc_term

def cosine_create_inv_idx(food_arr_uv):
#inv_idx[term] = [(d1, tf1), (d2, tf2), ...]    
    inv_idx = {}
    for i in range(len(food_arr_uv)):
        food_list, uv = food_arr_uv[i]
        for word in food_list:
            if word in inv_idx:
                inv_idx[word].append((i,uv))
            else:
                tf = math.log2(1 + max(0, uv))
                inv_idx[word] = [(i, tf)]
    return inv_idx

#compute idf values = 1/num docs containing term
def cosine_idf1(inv_idx):
    idf = {}
    for food, docs in inv_idx.items():
        idf[food] = 1/len(docs)
    return idf

def cosine_idf2(inv_idx,n_comms):
    idf2 = {}
    for food, docs in inv_idx.items():
        idf2[food] = math.log2(n_comms/(1+len(docs)))
    return idf2

def cosine_doc_norms(inv_idx, n_comms, idfs):
    norms = np.zeros(n_comms)
    for termi, idfi in idfs.items():
        for j, tfij in inv_idx.get(termi, []):
            norms[j] += (tfij * idfi) ** 2
    return np.sqrt(norms)

def cosine_dot_scores(query_word_counts,invidx,idfs):
    """
    returns a dict {docid : dot product sum}
    """
    acc  = {}
    for word, tfq in query_word_counts.items():
        for (dj,tfij) in invidx[word]:
            if dj not in acc.keys():
                acc[dj] = tfq*idfs[word] * (tfij * idfs[word])
            else:
                acc[dj] += tfq*idfs[word]  * (tfij* idfs[word])
    return acc


def cosine_sim(query,inv_idx,idfs,norms):
    res = []
    query = query.lower()
    query_terms = query.split(" ")
    qtf = {}
    for term in query_terms:
        if term in idfs.keys():
            if term not in qtf.keys():
                qtf[term] = 1
            else:
                qtf[term] += 1
    
    #compute query norm
    qnorm = 0
    for term, tfq in qtf.items():
        qnorm += (idfs[term]* tfq)**2

    print("qnorm:", qnorm)

    num = cosine_dot_scores(qtf,inv_idx,idfs)
    for i in range(np.size(norms)):
        dnorm = norms[i]
        if i in num.keys() and dnorm > 0:
            res.append((num[i]/(qnorm * dnorm),i))
    return sorted(res,reverse = True)

def main_cos(query, filtered_matrix, indices):
    food_arr_uv = []
    for foodPair, uv in foodPair_upvotes.items():
        food_arr = [f.lower().strip() for f in foodPair.split(",") if f.strip()]
        food_arr_uv.append((food_arr, uv))

    n_comms = len(foodPair_upvotes)
    inv_idx = cosine_create_inv_idx(food_arr_uv)
    idfs = cosine_idf2(inv_idx, n_comms)
    
    assert "cheese" in idfs

    norms = cosine_doc_norms(inv_idx, n_comms, idfs)

    query_terms = [term.strip().lower() for term in query.split(",") if term.strip()]
    query_text = " ".join(query_terms)
    #res = [(sim, doc id)] in decreasing order
    res = cosine_sim(query,inv_idx,idfs,norms)
    i=0
    if len(res) <1:
        print("Try searching for another food item")
    while i < 3 and i < len(res) :
        print(i+1, food_arr_uv[res[i][1]][0])
        i+=1

    # << changes >> 
    cosine_score_vec = np.zeros(len(filtered_matrix))
    for score, doc_id in res: # boolean results apply to cosine as well
        if doc_id in indices:
            idx_in_filtered = np.where(indices == doc_id)[0][0]
            food_len = len(food_arr_uv[doc_id][0])
            penalty = 1 / math.sqrt(food_len)  # really small penalty for length
            cosine_score_vec[idx_in_filtered] = score * penalty


    max_value = cosine_score_vec.max()
    if max_value > 0:
        cosine_score_vec /= max_value

    return cosine_score_vec