import pickle 

vocab = []

with open("foodVocab.txt", "r") as f: 
  rawTxt = f.read().lower()
  vocab = rawTxt.split(", ")

vocab = set(vocab)

with open("tempDataset.pkl", "rb") as file:
    comm_score_dict = pickle.load(file)

#print(type(comm_score_dict))  
#print(len(comm_score_dict)) 

foodComm_score_dict = {}

for comm, score in comm_score_dict.items(): 
  foodComm = ""
  for keyword in vocab: 
    if keyword in comm: 
      foodComm += str(keyword + ", ")
  if foodComm != "":
    foodComm_score_dict.update({foodComm: score}) 

print(foodComm_score_dict)
print(len(foodComm_score_dict))

with open("tempPreprocessed.txt", "w", encoding='utf-8', errors='ignore') as f: 
  f.write(str(foodComm_score_dict))

with open("tempPreprocessed.pkl", "wb") as file:
  pickle.dump(foodComm_score_dict, file)


    
    

   





