import os 
import pickle
import numpy as np
from numpy.typing import NDArray 
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "python_scripts_and_data", "data")

# Load vocab
with open(os.path.join(DATA_DIR, "foodVocab.pkl"), "rb") as file: 
    vocab = pickle.load(file)

# Load complexRep
with open(os.path.join(DATA_DIR, "complexRep.pkl"), "rb") as file:
    complexRep = pickle.load(file)

# Load flavors
with open(os.path.join(DATA_DIR, "flavors.pkl"), "rb") as file:
    flavors = pickle.load(file)

#fakeData = np.zeros((10, len(flavors)))

food_desc_dict1 = { 
    "tuna" : ["silky", "rich", "fatty"], 
    "bagel" : ["distinctive", "chewy", "crisp", "yeasty"], 
    "honey" : ["sweet", "sugary", "floral", "robust", "complex"], 
    "pear" : ["crisp", "buttery", "sweet", "juicy", "crunchy", "soft", "buttery"], 
    "pancakes" : ["sweet", "soft", "fluffy", "spongy", "yeasty"], 
    "cardamom" : ["sweet", "warm", "floral", "spicy", "mint", "warm", "smoky", "earthy", "citrusy"], 
    "flax seeds" : ["nutty", "earthy", "toasty", "sweet", "earthy"], 
    "tiramisu" : ["rich", "creamy", "sweet", "bitter", "robust"], 
    "stevia" : ["sweet", "bitter", "licorice", "sugary", "metallic"], 
    "broccoli" : ["bitter", "earthy", "vegetal", "sweet", "crunchy", "crisp", "tender"]
}

food_desc_dict2 = {
    "tuna" : "silky rich fatty", 
    "bagel" : "distinctive chewy crisp yeasty", 
    "honey" : "sweet sugary floral robust complex", 
    "pear" : "crisp buttery sweet juicy crunchy soft buttery", 
    "pancakes" : "sweet soft fluffy spongy yeasty", 
    "cardamom" : "sweet warm floral spicy mint warm smoky earthy citrusy", 
    "flax seeds" : "nutty earthy toasty sweet earthy", 
    "tiramisu" : "rich creamy sweet bitter robust", 
    "stevia" : "sweet bitter licorice sugary metallic", 
    "broccoli" : "bitter earthy vegetal sweet crunchy crisp tender"
}

###format data similar to demo

#documents1 = [list(food_desc_dict2.values())]
#print(documents1)

documents2 = list(food_desc_dict2.values())
print(documents2)

vectorizer = TfidfVectorizer(vocabulary=flavors)
td_matrix = vectorizer.fit_transform([x for x in documents2])

print(type(td_matrix))
#print(td_matrix)
print(td_matrix.shape)
#print(td_matrix.transpose().shape)

from scipy.sparse.linalg import svds

u, s, v_trans = svds(td_matrix, k=9) #k value wth value of td_matrix.shape
print(u.shape)
print(s.shape)
print(v_trans.shape)

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.plot(s[::-1])
plt.xlabel("Singular value number")
plt.ylabel("Singular value")
plt.show()

docs_compressed, s, words_compressed = svds(td_matrix, k=7)
words_compressed = words_compressed.transpose()

print(docs_compressed.shape)
print(words_compressed.shape)


word_to_index = vectorizer.vocabulary_
index_to_word = {i:t for t,i in word_to_index.items()}

#print(index_to_word)

#row normalize
from sklearn.preprocessing import normalize
words_compressed_normed = normalize(words_compressed, axis = 1)
#print(type(words_compressed_normed))


# cosine similarity
def closest_words(word_in, words_representation_in, k = 5):
    if word_in not in flavors: return "Flavor Not in vocab."
    sims = np.dot(words_representation_in, (words_representation_in[word_to_index[word_in],:]))
    asort = np.argsort(-sims)[:k+1]
    return [(index_to_word[i],sims[i]) for i in asort[1:]]

td_matrix_np = td_matrix.transpose().toarray()
td_matrix_np = normalize(td_matrix_np)
#print(type(td_matrix_np))

word = 'sugary'
retVal = closest_words(word, words_compressed_normed)
if isinstance(retVal, str): 
    print(retVal)
else: 
    for (w, sim) in retVal: 
        print("{}, {:.3f}".format(w, sim))
  
print()
print("Without using SVD: using term-doc matrix directly:")
retVal = closest_words(word, td_matrix_np)
if isinstance(retVal, str): 
    print(retVal)
else: 
    for (w, sim) in retVal: 
        print("{}, {:.3f}".format(w, sim))
  
print()

print("Tuna:\t" + str(docs_compressed[0]))
print(np.dot(docs_compressed, docs_compressed[0].T))
print(np.argsort(np.dot(docs_compressed, docs_compressed[0].T)))
#print(np.argsort(-np.dot(td_matrix_np, td_matrix_np[0].T)))