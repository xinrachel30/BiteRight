import os 
import pickle
import numpy as np
from numpy.typing import NDArray 
from typing import Any

vocab = []

with open(os.path.join("data", "foodVocab.txt"), "r") as f: 
  rawTxt = f.read().lower()
  vocab = rawTxt.split(", ")

vocab = list(set(vocab))

with open(os.path.join("data", "foodwords_score_dict.pkl"), "rb") as file:
    docContent_upvote_dict = pickle.load(file)

def create_doc_term(foodPair_scores : dict[str,int], vocab: list[str], mode:str) -> NDArray[Any]:
  mode = mode.lower()
  doc_term_rep = np.zeros((len(foodPair_scores), len(vocab)))
  docIdx = 0
  for content, upvote in foodPair_scores.items(): 
    content_arr = content.split(", ")
    content_arr = content_arr[0:len(content_arr)-1] #Eliminates '' at end of arr
    #Invariant: Any term in content_arr is guarenteed to be in vocab
    for term in content_arr: 
      tIdx = vocab.index(term)
      if mode == "bin": 
        doc_term_rep[docIdx][tIdx] = 1
      elif mode == "tf":
        doc_term_rep[docIdx][tIdx] = upvote
      else: 
        raise Exception("mode must only equal \"bin\" or \"tf\"")
    docIdx += 1
  return doc_term_rep

#Both Jaccard Functions return a (n,) ndarray where n is the number of docs
#No difference between these two because of way data is formatted in sprint 1
#gen_jaccard is probably implemented correctly
def set_jaccard_sim(query, doc_term_mat):
  jaccard_result = np.zeros((len(doc_term_mat), )) 
  doc_term_mat = np.where(doc_term_mat > 0, 1, 0)
  query = np.where(query > 0, 1, 0)
  qNdt = np.dot(doc_term_mat, query.T)
  query_expand = np.tile(query, (len(doc_term_mat), 1))
  qUdt = np.where((doc_term_mat+query_expand) > 0, 1, 0)
  qUdt = np.sum(qUdt, axis=1) #Sum Rowwise

  jaccard_result = qNdt/qUdt
  return jaccard_result

def gen_jaccard_sim(query, doc_term_mat): 
  jaccard_result = np.zeros((len(doc_term_mat), )) 

  w_query = query / np.sum(query)

  w_doc_term_mat = np.zeros(doc_term_mat.shape)
  for rIdx in range(0, len(doc_term_mat)):
    w_row = doc_term_mat[rIdx] / np.sum(doc_term_mat[rIdx])
    w_doc_term_mat[rIdx] = w_row

  w_query_expand = np.tile(w_query, (len(w_doc_term_mat), 1))
  termWeights = np.minimum(w_query_expand, w_doc_term_mat) / np.maximum(w_query_expand, w_doc_term_mat)

  query_expand = np.tile(query, (len(doc_term_mat), 1))
  qUdtMask = np.where((doc_term_mat+query_expand) > 0, 1, 0)

  jaccard_result_expand = np.where(qUdtMask > 0, termWeights, 0)

  jaccard_result = np.sum(jaccard_result_expand, axis=1)

  return jaccard_result


# doc_term_bin_rep = create_doc_term(docContent_upvote_dict, vocab, mode="bin")
# doc_term_tf_rep = create_doc_term(docContent_upvote_dict, vocab, mode="tf")

# sample_query = np.zeros((len(vocab),))
# sample_query[0] = 1
# sample_query[10] = 20
# sample_query[50] = 16

# results1 = set_jaccard_sim(sample_query, doc_term_bin_rep)
# results2 = set_jaccard_sim(sample_query, doc_term_tf_rep)

# results3 = gen_jaccard_sim(sample_query, doc_term_bin_rep)
# results4 = gen_jaccard_sim(sample_query, doc_term_tf_rep)

# print(results1)
# print(results2)
# print(results3)
# print(results4)