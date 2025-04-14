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
  vocab = list(set(vocab))
  with open(os.path.join("data", "foodVocab.pkl"), "wb") as file: 
    pickle.dump(vocab, file)

def flavors_preprocess(): 
  flavors = []
  with open(os.path.join("data", "flavors.txt"), "r") as f: 
    rawTxt = f.read().lower()
    flavors = rawTxt.split(", ")
  #print(len(flavors))
  #print(str(flavors))
  flavors = list(set(flavors))
  #print(len(flavors))
  #print(str(flavors))
  with open(os.path.join("data", "flavors.pkl"), "wb") as file: 
    pickle.dump(flavors, file)

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

def complexCommentRep(vocab):
  '''
  For all unique comments captured in the dataset, returns this representation: 
  {commentId : ({Food1 : count of occurence of Food1 in comment}, upvotes of commment)}

  This method does not filter out comment without food words in vocaulary, but
  such a case would appear as this: {int x: ({}, int y)}
  where x is the id and y is the upvote score. 
  '''

  with open(os.path.join("data", "comment_score_dict.pkl"), "rb") as file:
    comm_score_dict = pickle.load(file)
  complexRep = {}
  commentId = 0
  for comment, upvote in comm_score_dict.items(): 
    nestedDict = {}
    for food in vocab: 
      f_count = comment.count(food)
      if f_count > 0: 
        nestedDict.update({food:f_count})
    complexRep.update({commentId:(nestedDict, upvote)})
    commentId += 1
  with open(os.path.join("data", "complexRep.txt"), "w", encoding='utf-8', errors='ignore') as f: 
    f.write(str(complexRep))
  with open(os.path.join("data", "complexRep.pkl"), "wb") as file:
    pickle.dump(complexRep, file)

def food_flavor_data_preprocess(vocab, flavors): 
  with open(os.path.join("data", "sweet.txt"), "r") as file:
    sweetFoods = [line.strip().lower() for line in file if line.strip()]
  with open(os.path.join("data", "spicy.txt"), "r") as file:
    spicyFoods = [line.strip().lower() for line in file if line.strip()]
  with open(os.path.join("data", "fried.txt"), "r") as file:
    friedFoods = [line.strip().lower() for line in file if line.strip()]
  with open(os.path.join("data", "cold.txt"), "r") as file:
    coldFoods = [line.strip().lower() for line in file if line.strip()]

  sweetIdx = flavors.index("sweet")
  spicyIdx = flavors.index("spicy")
  coldIdx = flavors.index("cold")
  friedIdx = flavors.index("fried")

  food_flavor_mat = np.zeros((len(vocab), len(flavors)))
  for flavFood in sweetFoods: 
    for foodIdx in range(0, len(vocab)): 
      if vocab[foodIdx] in flavFood:
        food_flavor_mat[foodIdx][sweetIdx] = 1
        #print(str(vocab[foodIdx]) + " is sweet") 

  for flavFood in spicyFoods: 
    for foodIdx in range(0, len(vocab)): 
      if vocab[foodIdx] in flavFood:
        food_flavor_mat[foodIdx][spicyIdx] = 1

  for flavFood in coldFoods: 
    for foodIdx in range(0, len(vocab)): 
      if vocab[foodIdx] in flavFood:
        food_flavor_mat[foodIdx][coldIdx] = 1

  for flavFood in friedFoods: 
    for foodIdx in range(0, len(vocab)): 
      if vocab[foodIdx] in flavFood:
        food_flavor_mat[foodIdx][friedIdx] = 1

  #Formatted as in demo for unsupervised learning
  foodFlavorsList = []
  for i in range(0, len(food_flavor_mat)): 
    food_flavor_str = ""
    if food_flavor_mat[i][sweetIdx] == 1:
      food_flavor_str += "sweet "
    if food_flavor_mat[i][spicyIdx] == 1: 
      food_flavor_str += "spicy "
    if food_flavor_mat[i][coldIdx] == 1: 
      food_flavor_str += "cold "
    if food_flavor_mat[i][friedIdx] == 1: 
      food_flavor_str += "fried "
    foodFlavorsList.append(food_flavor_str)

  with open(os.path.join("data", "food_flavors_data.txt"), "w", encoding='utf-8', errors='ignore') as f: 
    f.write(str(foodFlavorsList))
  with open(os.path.join("data", "food_flavors_data.pkl"), "wb") as file:
    pickle.dump(foodFlavorsList, file)
  

vocab_preprocess()
flavors_preprocess()
with open(os.path.join("data", "foodVocab.pkl"), "rb") as file: 
  vocab = pickle.load(file)
isolateFoodwords(vocab)
with open(os.path.join("data", "foodwords_score_dict.pkl"), "rb") as file: 
  foodwords_score_dict = pickle.load(file)
complexCommentRep(vocab)
with open(os.path.join("data", "complexRep.pkl"), "rb") as file: 
  complexRep = pickle.load(file)
with open(os.path.join("data", "flavors.pkl"), "rb") as file: 
  flavors = pickle.load(file)
food_flavor_data_preprocess(vocab, flavors)
    
    

   





