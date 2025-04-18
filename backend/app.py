import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from python_scripts_and_data.jaccard_sim import *
from python_scripts_and_data.jaccard_sim import create_doc_term
from python_scripts_and_data.unsupervised import *
from python_scripts_and_data.evaluation import *
import numpy as np

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
LOCAL_MYSQL_USER = "wsl_root"
LOCAL_MYSQL_USER_PASSWORD = "admin" # TODO: make this an env variable
LOCAL_MYSQL_PORT = 3306
LOCAL_MYSQL_DATABASE = "biterightdb"

mysql_engine = MySQLDatabaseHandler(LOCAL_MYSQL_USER,LOCAL_MYSQL_USER_PASSWORD,LOCAL_MYSQL_PORT,LOCAL_MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

# Sample search, the LIKE operator in this case is hard-coded, 
# but if you decide to use SQLAlchemy ORM framework, 
# there's a much better and cleaner way to do this
def sql_search(episode):
    query_sql = f"""SELECT * FROM episodes WHERE LOWER( title ) LIKE '%%{episode.lower()}%%' limit 10"""
    keys = ["id","title","descr"]
    data = mysql_engine.query_selector(query_sql)
    return json.dumps([dict(zip(keys,i)) for i in data])

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return sql_search(text)

@app.route("/flavor-search")
def flavor_search(): 
    selected_flavors = request.args.getlist("flavors")
    ranking_dict = closest_flavor_calc(selected_flavors)
    ranking = list(ranking_dict.keys())

    # sims_scores contains the raw information for the entries
    # corresponding in ranking. Use as desired
    sims_scores = list(ranking_dict.values())
    
    return jsonify({"results": ranking})

@app.route('/search')
def search():
    query = request.args.get('query', '').lower()    
    if not query:
        return jsonify([])

    tokenized_query = tokenize_query(query.replace(',', '')) 
    # ex) "grape, pork and cheese" -> ['grape', 'pork', 'and', 'cheese]
    # ex) "(pork and, cheese)" -> ['(', 'pork', 'and', 'cheese', ')']
    print("tokenized:", tokenized_query)

    comma_separated = [section.strip() for section in query.split(',') if section.strip()]
    # ex) "cheese, pork and grape" -> ['cheese', 'pork and grape']

    cleaned_query_boolean = []
    for part in comma_separated: 
        tokens = tokenize_query(part)
        parsed = parse_parens(tokens)
        print("each part: ", part, "->", parsed)
        cleaned_query_boolean.append(parsed)
    # ex) ['cheese', 'pork and grape'] -> ['cheese', ['pork', 'and', 'grape'] ]
    
    final_boolean = []
    for i, parsed in enumerate(cleaned_query_boolean):
        if i > 0:
            final_boolean.append("or") # adds 'or' between every two "parts"
        final_boolean.append(parsed) 
    # ex) ['cheese', ['pork', 'and', 'grape'] ] -> ['cheese', 'or', ['pork', 'and', 'grape'] ]

    # Initialize variables
    query_vector = np.zeros(len(vocab))
    typo_suggestions = []
    contains_booleans = False

    # For creating query vector, modified to also find suggested words
    for term in tokenized_query:
        if term in vocab:
            idx = vocab.index(term)
            query_vector[idx] += 1
            print(term, "is a vocab word")
        elif (not contains_booleans) and term in ["or", "and", "not", "(", ")"]: #considering parentheses as booleans
            contains_booleans = True
            print("contains booleans")
        else:
            print("needed to find a suggestion for", term)
            typo_suggestions.extend(find_closest(term, vocab)) #if misspelt, find typos suggestions

    query_vocab_terms = [term for term in tokenized_query if term in vocab]
    print("final list of vocab: ", query_vocab_terms)
    
    print("typo suggestions: ", typo_suggestions)
    if not query_vocab_terms: #no vocab words 
        return jsonify({
            "results": [],
            "suggestions": typo_suggestions
        })

    # Create document-term matrix
    doc_term_matrix = create_doc_term(complex_items, vocab, mode="tf")
    print("doc term matrix shape: ", doc_term_matrix.shape)
    doc_term_binary = np.where(doc_term_matrix > 0, 1, 0)
    print("number of documents: ", len(complex_items))

    if contains_booleans: 
        bool_mask = complete_boolean(cleaned_query_boolean, doc_term_binary, vocab, complexRep)
        print("contains booleans -> created bool_mask")
    else: 
        bool_mask = np.ones(len(complexRep), dtype=bool) #doesn't do anything
        print("bool mask is all ones")

    filtered_matrix = doc_term_matrix[bool_mask]
    print("filtered matrix shape:", filtered_matrix.shape)
    indices = np.where(bool_mask)[0]

    # Get similarity scores using generalized Jaccard
    similarity_scores = gen_jaccard_sim(query_vector, filtered_matrix)
    
    # Get top results where similarity > 0
    results = []
    already_seen = set() #remove duplicates
    
    for idx, score in enumerate(similarity_scores):
        if score > 0:
            true_idx = indices[idx]
            comment_id, (food_dict, _) = complex_items[true_idx]
            food_items = list(food_dict.keys())
            title = (", ".join(food_items)).strip()

            if title in already_seen or len(title) == len(food_items[0]): 
                continue
            else: 
                already_seen.add(title)

            results.append({
                'title': title,
                'similarity': float(score)
            })

    # Sort by similarity score descending
    results.sort(key=lambda x: x['similarity'], reverse=True)

    return jsonify({
        "results": results[:10], 
        "suggestions": typo_suggestions 
    })  # Return top 10 results


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)