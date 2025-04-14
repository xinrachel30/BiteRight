import os 
import pickle
import numpy as np
from numpy.typing import NDArray 
from typing import Any

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

def create_doc_term(complex_items, vocab: list[str], mode:str) -> NDArray[Any]: 
  mode = mode.lower()
  doc_term_rep = np.zeros((len(complex_items), len(vocab)))
  for docIdx, (comment_id, (innerDict, upvotes)) in enumerate(complex_items): 
    #Invariant: Any key of innerDict is guarenteed to be found in vocab
    #f_count_sum = sum(list(innerDict.values())) #Uncomment as needed
    for food, f_count in innerDict.items(): 
      tIdx = vocab.index(food)
      if mode == "bin": 
        doc_term_rep[docIdx][tIdx] = 1
      elif mode == "tf":
        # tf options: 
          # tf = upvotes * f_count (Choosing this for now)
          # tf = upvotes + f_count 
          # tf = upvotes * (f_count/f_count_sum)
        doc_term_rep[docIdx][tIdx] = upvotes * f_count
      else: 
        raise Exception("mode must only equal \"bin\" or \"tf\"")
  return doc_term_rep

#currently using substitution cost of 1, but may reconsider?
#maybe nearby letters can be substitution cost 1, but non-nearby is 2?
def edit_distance(word1, word2):
  x, y = len(word1), len(word2) #dimensions
  dp = [[0] * (y+1) for _ in range(x+1)] #number of rows is len(word1), number of cols is len(word2)

  for i in range(x + 1): 
    dp[i][0] = i
  for i in range(y + 1): 
    dp[0][i] = i
  
  for i in range(1, x+1): 
    for j in range(1, y+1): 
      if word1[i-1] == word2[j-1]: #same letter
        dp[i][j] = dp[i-1][j-1] #no diff
      else: 
        dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1 
  
  return dp[x][y]

def find_closest(term, vocab):
  typo_suggestions = []

  term = term.lower()
  min_distance = float('inf')
  second_min_distance = float('inf')
  closest_word = ""
  second_closest = ""

  for vocab_word in vocab:
      curr_distance = edit_distance(vocab_word, term)

      if curr_distance < min_distance:
          second_min_distance = min_distance
          second_closest = closest_word
          min_distance = curr_distance
          closest_word = vocab_word
      elif curr_distance < second_min_distance:
          second_min_distance = curr_distance
          second_closest = vocab_word
  typo_suggestions.append(closest_word)
  typo_suggestions.append(second_closest)  
  return typo_suggestions

def boolean_and(query_vec, doc_term_bin):
  has_query = doc_term_bin[:, query_vec == 1]
  results = np.all(has_query == 1, axis = 1)
  return results

def boolean_or(query_vec, doc_term_bin): 
  has_query = doc_term_bin[:, query_vec == 1]
  results = np.any(has_query == 1, axis = 1)
  return results

def construct_query_vec(query_words):
  query_vector = np.zeros((len(vocab), ))
  for word in query_words: 
    if word in vocab: 
      idx = vocab.index(word)
      query_vector[idx] = 1
  
  return query_vector

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

doc_term_bin_rep = create_doc_term(complex_items, vocab, mode="bin")
doc_term_tf_rep = create_doc_term(complex_items, vocab, mode="tf")

# sample_query = np.ones((len(vocab),))
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