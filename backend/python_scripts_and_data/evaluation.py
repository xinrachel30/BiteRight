import numpy as np
from python_scripts_and_data.jaccard_sim import *
from python_scripts_and_data.unsupervised import *
from python_scripts_and_data.evaluation import *

#currently using substitution cost of 1, but may reconsider?
#maybe nearby letters can be substitution cost 1, but non-nearby is 2?
def edit_distance(word1, word2):
  x, y = len(word1), len(word2) #dimensions
  dp = [[0] * (y+1) for _ in range(x+1)] #number of rows is len(word1), number of cols is len(word2)

  for i in range(x+1):
    dp[i][0] = i
  for i in range(y+1): 
    dp[0][i] = i
  
  for i in range(1, x+1): 
    for j in range(1, y+1): 
      if word1[i-1] == word2[j-1]: #same letter
        dp[i][j] = dp[i-1][j-1] #no diff
      else: 
        dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1 
  
  return dp[x][y]

def find_closest(term, term_list):
  typo_suggestions = []

  term = term.lower()
  min_distance = float('inf')
  second_min_distance = float('inf')
  closest_word = ""
  second_closest = ""

  for vocab_word in term_list:
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

def construct_query_vec(query_words, vocab):
  query_vector = np.zeros((len(vocab), ))
  for word in query_words: 
    if word in vocab: 
      idx = vocab.index(word)
      query_vector[idx] = 1
  
  return query_vector

def boolean_not(query_vec, doc_term_bin): 
    results = doc_term_bin[:, query_vec == 0]
    return results

def boolean_and(query_vec, doc_term_bin):
  has_query = doc_term_bin[:, query_vec == 1]
  results = np.all(has_query == 1, axis = 1)
  return results

def boolean_or(query_vec, doc_term_bin): 
  has_query = doc_term_bin[:, query_vec == 1]
  results = np.any(has_query == 1, axis = 1)
  return results

def tokenize_query(query):
    # ex) "pork and cheese" -> ['pork', 'and', 'cheese]
    # ex) "(pork and cheese)" -> ['(', 'pork', 'and', 'cheese', ')']
    tokens = []
    current_token = ""
    for char in query:
        if char in ["(", ")", " "]:
            if current_token:
                tokens.append(current_token)
                current_token = ""
            if char != " ":
                tokens.append(char)
        else:
            current_token += char
    if current_token:
        tokens.append(current_token)
    return tokens

tokenized = tokenize_query("cheese and (grape or lemon)")
print(tokenized)

def parse_parens(tokens):
    # ex) "(pork and cheese)" -> [ ['pork', 'and', 'cheese'] ]
    # ex) "(pork and cheese) or grape" -> [ ['pork', 'and', 'cheese'], 'or', 'grape' ]
    stack = []
    curr = []

    for token in tokens:
        if token == "(":
            stack.append(curr)
            curr = []
        elif token == ")":
            subquery = curr
            curr = stack.pop()
            curr.append(subquery)
        else:
            curr.append(token)
    return curr

the_best_query = parse_parens(tokenized)
print(the_best_query)

#pre-condition: query does not have commas, should have already ran tokenize and parse
def complete_boolean(query, doc_term_bin, vocab, complexRep):
    try:
        if isinstance(query, str):
            query_vec = construct_query_vec([query], vocab)
            return boolean_or(query_vec, doc_term_bin)
        if isinstance(query, list) and len(query) == 1:
            return complete_boolean(query[0], doc_term_bin, vocab, complexRep)
        
        i = 0
        current_mask = None
        while i < len(query):
            token = query[i]

            if token == "not": 
                i += 1
                if i >= len(query): 
                    return np.ones(len(complexRep), dtype=bool)
                negated = complete_boolean(query[i], doc_term_bin, vocab, complexRep)
                sub_mask = np.logical_not(negated)
                if current_mask is None:
                    current_mask = sub_mask
                else: 
                    current_mask = np.logical_and(current_mask, sub_mask)
            elif token == "and" or token == "or":
                op = token
                i += 1
                if i >= len(query):
                    np.ones(len(complexRep), dtype=bool)
                right = query[i]
                right_mask = complete_boolean(right, doc_term_bin, vocab, complexRep)

                if op == "and":
                    current_mask = np.logical_and(current_mask, right_mask)
                elif op == "or":
                    current_mask = np.logical_or(current_mask, right_mask)
            else:
                subquery = token
                sub_mask = complete_boolean(subquery, doc_term_bin, vocab, complexRep)
                if current_mask is None:
                    current_mask = sub_mask
                else:
                    current_mask = np.logical_and(current_mask, sub_mask)
            i += 1

        return current_mask if current_mask is not None else np.ones(len(complexRep), dtype=bool)

    except Exception as e: # the main case this happens is when the user improperly formats
        #may want to return some indication of this happening to app.py
        return np.ones(doc_term_bin.shape[0], dtype=bool)

# for i, doc in enumerate(matched_docs):
#     print(f"doc{i}, things in doc:{doc}")

# vocab = ["cheese", "grape", "lemon", "banana", "cream", "pork"]
# doc_term_bin = np.array([
#     [1, 1, 0, 0, 0, 0],  # doc 0: cheese, grape
#     [0, 0, 0, 0, 1, 0],  # doc 1: cream
#     [1, 0, 0, 1, 0, 0],  # doc 2: cheese, banana
#     [1, 1, 1, 1, 0, 0],  # doc 3: cheese, grape, lemon, banana
#     [0, 0, 0, 0, 1, 1],  # doc 4: cream, pork
#     [1, 0, 0, 0, 0, 1],  # doc 5: cheese, pork
# ])
# complex_items = list(enumerate([({}, "") for _ in doc_term_bin]))
# complexRep = [None] * len(doc_term_bin)  # just dummy placeholders

# def print_test(query):
#     mask = complete_boolean(query, doc_term_bin, vocab, complexRep)
#     matched = [i for i, val in enumerate(mask) if val]
#     print("Query:", query)
#     print("Matched indices :", matched)

# print_test(['cheese', 'and', 'grape'])

# print_test(['banana', 'or', 'cream'])

