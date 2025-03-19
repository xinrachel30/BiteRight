import pickle 
import os
import numpy as np
from numpy.typing import NDArray 
from typing import Any
from collections import OrderedDict

def vocab_preprocess(): 
  vocab = []
  with open(os.path.join("data", "foodVocab.txt"), "r") as f: 
    rawTxt = f.read().lower()
    vocab = rawTxt.split(", ")
  vocab = set(vocab)
  with open(os.path.join("data", "foodVocab.pkl"), "wb") as file: 
    pickle.dump(vocab, file)

def isolateFoodwords(vocab): 
  with open(os.path.join("data", "comment_score_dict.pkl"), "rb") as file:
    comm_score_dict = pickle.load(file)
  foodComm_score_dict = {}
  for comm, score in comm_score_dict.items(): 
    foodComm = ""
    for keyword in vocab: 
      if keyword in comm: 
        foodComm += str(keyword + ", ")
    # if foodComm != "":   temporary testing!
      foodComm_score_dict.update({foodComm: score}) 
  with open(os.path.join("data", "foodwords_score_dict.txt"), "w", encoding='utf-8', errors='ignore') as f: 
    f.write(str(foodComm_score_dict))
  with open(os.path.join("data", "foodwords_score_dict.pkl"), "wb") as file:
    pickle.dump(foodComm_score_dict, file)

vocab_preprocess()
with open(os.path.join("data", "foodVocab.pkl"), "rb") as file: 
  vocab = pickle.load(file)
isolateFoodwords(vocab)
with open(os.path.join("data", "foodwords_score_dict.pkl"), "rb") as file: 
  foodwords_score_dict = pickle.load(file)
    
    

   





