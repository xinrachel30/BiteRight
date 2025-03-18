import os 
import pickle
import numpy as np

vocab = []

with open(os.path.join("data", "foodVocab.txt"), "r") as f: 
  rawTxt = f.read().lower()
  vocab = rawTxt.split(", ")

vocab = list(set(vocab))

vocab_dict = {keyword: 0 for keyword in vocab}

with open(os.path.join("data", "foodwords_score_dict.pkl"), "rb") as file:
    docContent_upvote_dict = pickle.load(file)

doc_term_bin_rep = np.zeros((len(docContent_upvote_dict), len(vocab)))

#Using upvote as the tf
doc_term_tf_rep = np.zeros((len(docContent_upvote_dict), len(vocab))) 

docIdx = 0
for content, upvote in docContent_upvote_dict.items(): 
  content_arr = content.split(", ")
  content_arr = content_arr[0:len(content_arr)-1] #Eliminates '' at end of arr
  #Invariant: Any term in content_arr is guarenteed to be in vocab
  for term in content_arr: 
    tIdx = vocab.index(term)
    doc_term_bin_rep[docIdx][tIdx] = 1
    doc_term_tf_rep[docIdx][tIdx] = upvote
  docIdx += 1

# Current rep dimensions: 
# N x M
# where N is number of documents
# where M is number of terms in vocabulary

# A query is size 1 x M

# Answer returned is size N x 1 
# Allows for us rank based on best score
# Returning from docContent_upvote_dict allows us to return just the foods
# Alternatively, could return most relevant comments in their entirety

def jaccard_sim(query, doc_term_mat): 
   pass