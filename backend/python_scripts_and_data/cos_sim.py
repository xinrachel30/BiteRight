import os 
import pickle
import numpy as np
import math

vocab = []

with open(os.path.join("backend/python_scripts_and_data/data", "foodVocab.txt"), "r") as f: 
  rawTxt = f.read().lower()
  vocab = rawTxt.split(", ")

vocab = list(set(vocab))
vocab_dict = {}
i = 0
for food in vocab:   
    vocab_dict[food] = i
    i += 1

with open(os.path.join("backend/python_scripts_and_data/data", "foodwords_score_dict.pkl"), "rb") as file:
    foodPair_upvotes = pickle.load(file)

def create_doc_term(foodPair_upvotes):
    doc_term = np.zeros((len(foodPair_upvotes),len(vocab)))
    # term doc matrix with tf = upvotes
    docIdx = 0
    for foodPair, uv in foodPair_upvotes.items():
        food_arr = foodPair.split(", ")
        food_arr = food_arr[0:len(food_arr)-1]
        for food in food_arr:
            doc_term[docIdx][vocab_dict[food]] = uv
        docIdx+=1
    return doc_term

def create_inv_idx(food_arr_uv):
#inv_idx[term] = [(d1, tf1), (d2, tf2), ...]    
    inv_idx = {}
    for i in range(len(food_arr_uv)):
        for word in vocab:
            if word in food_arr_uv[i][0]:
                if word in inv_idx:
                    inv_idx[word].append((i,uv))
                else:
                    inv_idx[word] = [(i,uv)]
    return inv_idx

#compute idf values = 1/num docs containing term
def idf1(inv_idx):
    idf = {}
    for food, docs in inv_idx.items():
        idf[food] = 1/len(docs)
    return idf

def idf2(inv_idx,n_comms):
    idf2 = {}
    for food, docs in inv_idx.items():
        idf2[food] = math.log2(n_comms/(1+len(docs)))
    return idf2

# compute doc norms
def doc_norms(inv_idx,n_comms):
    norms = np.zeros(n_comms)
    for termi, idfi in idf.items():
            for j, tfij in inv_idx[termi]:
                norms[j] += (tfij * idfi)**2
    return np.sqrt(norms)

def dot_scores(query_word_counts,invidx,idfs):
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
    qnorm = math.sqrt(qnorm)
    num = dot_scores(qtf,inv_idx,idfs)
    for i in range(np.size(norms)):
        dnorm = norms[i]
        if i in num.keys():
            res.append((num[i]/(qnorm * dnorm),i))
    return sorted(res,reverse = True)

def main(query):
    with open(os.path.join("backend/python_scripts_and_data/data", "foodwords_score_dict.pkl"), "rb") as file:
        foodPair_upvotes = pickle.load(file)
    
    #make a list of tuples [([food words],upvotes), (lst,int)]
    food_arr_uv = []
    for foodPair, uv in foodPair_upvotes.items():
        temp = foodPair.split(", ")
        food_arr = temp[0:len(temp)-1]
        food_arr_uv.append((food_arr,uv))

    n_comms = len(foodPair_upvotes)
    inv_idx = create_inv_idx(food_arr_uv)
    idfs = idf2(inv_idx,n_comms)
    norms = doc_norms(inv_idx,n_comms)
    #res = [(sim, doc id)] in decreasing order
    res = cosine_sim(query,inv_idx,idfs,norms)
    i=0
    if len(res) <1:
        print("Try searching for another food item")
    while i < 3 and i < len(res) :
        print(i+1, food_arr_uv[res[i][1]][0])
        i+=1
    return res

