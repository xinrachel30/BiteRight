import os 
import pickle
import numpy as np
import math

# Update paths to use the correct directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "python_scripts_and_data", "data")

# Load vocab
with open(os.path.join(DATA_DIR, "foodVocab.pkl"), "rb") as file: 
    vocab = pickle.load(file)

# Load complexRep
with open(os.path.join(DATA_DIR, "complexRep.pkl"), "rb") as file:
    complexRep = pickle.load(file)
    complex_items = list(complexRep.items())

vocab = list(set(vocab))
vocab_dict = {}
i = 0
for food in vocab:   
    vocab_dict[food] = i
    i += 1

def cos_create_doc_term(foodPair_upvotes):
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

def create_inv_idx(complex_items):
    # should use
    # complex items: [(comment#,({food: count},upvotes)),(int,(dict,int))]
    #inv_idx[term] = [(d1, tf1), (d2, tf2), ...]    
    inv_idx = {}
    for i in range(len(complex_items)):
        uv = complex_items[i][1][1]
        for word in vocab:
            if word in complex_items[i][1][0].keys():
                if word in inv_idx.keys():
                    inv_idx[word].append((complex_items[i][0],complex_items[i][1][0][word]*uv))
                else:
                    inv_idx[word] = [(complex_items[i][0],complex_items[i][1][0][word]*uv)]
    return inv_idx


#compute idf values = 1/num docs containing term
def idf1(inv_idx):
    idf = {}
    for food, docs in inv_idx.items():
        idf[food] = 1/len(docs)
    return idf

def idf2(inv_idx,n_comms):
    #inv_idx[term] = [(d1, tf1), (d2, tf2), ...]  
    idf2 = {}
    for food, docs in inv_idx.items():
        idf2[food] = math.log2(n_comms/(1+len(docs)))
    return idf2

# compute doc norms
def doc_norms(inv_idx,n_comms,idf):
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


def cosine_sim(query,inv_idx,idfs,norms,n_comms):
    """query vector is len(vocab) with tf"""
    
    # updated to return list of scores for each doc
    
    res = np.zeros(n_comms,dtype=float)
    qtf = {}
    for i in range(len(query)):
        if query[i] > 0:
            qtf[vocab[i]] = query[i]

    qnorm = np.linalg.norm(query)

    num = dot_scores(qtf,inv_idx,idfs)

    for i in range(n_comms):
        dnorm = norms[i]
        if i in num.keys():
            print(qnorm,"d:",dnorm)
            res[i] = num[i]/(qnorm*dnorm)
    return res


def main_cos(query,filtered_matrix, indices):

    n_comms = len(complex_items)
    inv_idx = create_inv_idx(complex_items)
    idfs = idf2(inv_idx,n_comms)
    norms = doc_norms(inv_idx,n_comms,idfs)
    sim_scores = cosine_sim(query,inv_idx,idfs,norms,n_comms)
    penalties = np.zeros(n_comms)
    for i in range(n_comms):
        if len(complex_items[i][1][0])>0:
            penalties[i] = (1e9) / math.sqrt(len(complex_items[i][1][0])) # really small penalty for length
    
    sim_scores *= penalties
    max = np.nanmax(sim_scores)
    if max > 0:
        sim_scores /= max
    return sim_scores

"""
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
"""