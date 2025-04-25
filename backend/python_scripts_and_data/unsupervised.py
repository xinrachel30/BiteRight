import os 
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


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

with open(os.path.join(DATA_DIR, "food_flavors_data.pkl"), "rb") as file: 
    #Assumption: Shape is (n,) where is number of foods recognized in vocab
    ff_data = pickle.load(file)

###Declaration of the vectorizer is essential
vectorizer = TfidfVectorizer(vocabulary=flavors)
#n x m, where n is as above and m is number of recognized flavors
food_flavor_matrix = vectorizer.fit_transform([x for x in ff_data])

#Note: k is limited: if food_flavor_matrix.shape = (n, m)
#k = min(n,m) - 1 at it's largest
#print(food_flavor_matrix.shape)
u, s, v_trans = svds(food_flavor_matrix, k=100)

###This can be removed once we select an appropriate value of k
plt.plot(s[::-1])
plt.xlabel("Singular value number")
plt.ylabel("Singular value")
# plt.show()  #Commented out once an appropriate k is chosen

'''
Appropriate Value of k is set here (Affects number of latent flavor dims)
foods_compressed is (n x k) --> See food latent dim values 
flavors compressed.T is (m x k) --> See flavor latent dim values 
s is a (k x k) byproduct --> It's useless. 
'''
foods_compressed, s, flavors_compressed = svds(food_flavor_matrix, k=70)
flavors_compressed = flavors_compressed.transpose()

#Normalization step can occur here 
foods_compressed_norm = normalize(foods_compressed)
flavors_compressed_norm = normalize(flavors_compressed, axis=1) #row-wise because transposed

with open(os.path.join(DATA_DIR, "unsupervisedData.pkl"), "wb") as file:
    pickle.dump((foods_compressed_norm, flavors_compressed_norm), file)
#Above Steps should only happen once in prepreprocessing
#Separating and putting steps here for now

with open(os.path.join(DATA_DIR, "unsupervisedData.pkl"), "rb") as file: 
  (foods_compressed_norm, flavors_compressed_norm) = pickle.load(file)

'''
Requires food_list to be preprocessed:
> Strings are in lowercase
> all elements of food_list are in vocab

Returns an ordering of foods closest to flavor profile
'''
def closest_food_profile(food_list, food_latent_rep=foods_compressed_norm): 
    query = np.zeros((len(food_latent_rep[0]),))
    for food in food_list: 
        foodIdx = vocab.index(food)
        query += food_latent_rep[foodIdx]
    sims = np.dot(food_latent_rep, query.T)
    desc_order = np.argsort(sims)[::-1]
    closest_foods = {}
    for i in desc_order:
        if vocab[i] in food_list: continue
        closest_foods.update({vocab[i]: sims[i]})
        if len(closest_foods) == 10: 
            break 
    return closest_foods

'''
Requires flavor_list to be preprocessed:
> Strings are in lowercase
> all elements of flavor_list are in recognized flavors

Returns an ordering of foods closest to flavor profile
'''
def closest_flavor_calc(flavor_list, food_latent_rep=foods_compressed_norm):
    flav_str = ""
    for flav in flavor_list: 
        flav_str += (flav + " ")
    query = vectorizer.transform([flav_str]).toarray()
    query_vec = np.dot(query, flavors_compressed)
    query_norm = normalize(query_vec).flatten()
    sims = np.dot(food_latent_rep, query_norm.T)
    desc_order = np.argsort(sims)[::-1]
    closest_foods = [(vocab[i], sims[i]) for i in desc_order]
    closest_foods = closest_foods[:10]
    closest_foods = dict(closest_foods)
    return closest_foods

def closest_flavors_given_foods(result_foods, food_latent_rep=foods_compressed_norm, flavor_latent_rep=flavors_compressed_norm):
    query = np.zeros((food_latent_rep.shape[1],))
    
    for food in result_foods:
        food_idx = vocab.index(food)
        query += food_latent_rep[food_idx]
    
    query = normalize(query.reshape(1, -1)).flatten()
    sims = np.dot(flavor_latent_rep, query.T)
    desc_order = np.argsort(sims)[::-1]
    top_flavors = [(flavors[i], sims[i]) for i in desc_order[:10]]

    return dict(top_flavors)


print("Tastes like an apple")
print(str(list(closest_food_profile(["apple"], foods_compressed_norm).keys())[:10]))
print("\n")

print("Tastes like an apple and ice cream")
print(str(list(closest_food_profile(["apple", "ice cream"], foods_compressed_norm).keys())[:10]))
print("\n")

print("Tastes like spaghetti, apple, and cilantro")
print(str(list(closest_food_profile(["spaghetti", "apple", "cilantro"], foods_compressed_norm).keys())[:10]))
print("\n")

print("Tastes sweet")
print(str(list(closest_flavor_calc(["sweet"], foods_compressed_norm).keys())[:10]))
print("\n")

print("Apple and peach")
print((closest_flavors_given_foods(["apple", "peach"])["sugary"]))
print("\n")

print("Tastes sweet and spicy")
print(str(list(closest_flavor_calc(["sweet", "spicy"], foods_compressed_norm).keys())[:10]))
print("\n")

print("Tastes robust, sugary, greasy, and rich")
print(str(list(closest_flavor_calc(["robust", "sugary", "greasy", "rich"], foods_compressed_norm).keys())[:10]))
print("\n")