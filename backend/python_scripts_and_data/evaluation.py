import numpy as np

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

def parse_parens(tokens):
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

#pre-condition: there are no commas in query_vec, lowercase
def iterative_boolean(query, doc_term_bin, vocab):
    def evaluate_subquery(subquery, doc_term_bin, vocab):
        shortened_query_vec = subquery[:]
        new_docs = doc_term_bin
        while len(shortened_query_vec) >= 2:
            if shortened_query_vec[0] == "not":
                if shortened_query_vec[1] in vocab:
                    new_docs = boolean_not(shortened_query_vec, new_docs)
                shortened_query_vec.pop(0)
                shortened_query_vec.pop(0)
            elif len(shortened_query_vec) >= 3:
                if shortened_query_vec[1] == "and":
                    if shortened_query_vec[0] in vocab and shortened_query_vec[2] in vocab:
                        new_docs = boolean_and(shortened_query_vec, new_docs)
                    shortened_query_vec.pop(0)
                    shortened_query_vec.pop(0)
                elif shortened_query_vec[1] == "or":
                    if shortened_query_vec[0] in vocab and shortened_query_vec[2] in vocab:
                        new_docs = boolean_or(shortened_query_vec, new_docs)
                    shortened_query_vec.pop(0)
                    shortened_query_vec.pop(0)
        return new_docs

    stack = []
    current_query = []
    for token in query:
        if token == "(":
            stack.append(current_query)
            current_query = []
        elif token == ")":
            subquery_docs = evaluate_subquery(current_query, doc_term_bin, vocab)
            current_query = stack.pop()
            current_query.append(subquery_docs)
        else:
            current_query.append(token)
    
    return evaluate_subquery(current_query, doc_term_bin, vocab)



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

def construct_query_vec(query_words):
  query_vector = np.zeros((len(vocab), ))
  for word in query_words: 
    if word in vocab: 
      idx = vocab.index(word)
      query_vector[idx] = 1
  
  return query_vector