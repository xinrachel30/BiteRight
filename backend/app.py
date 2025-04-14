import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from python_scripts_and_data.jaccard_sim import *
from python_scripts_and_data.unsupervised import *
import numpy as np

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
LOCAL_MYSQL_USER = "root"
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
    ranking = closest_flavor_calc(selected_flavors)
    return jsonify({"results": ranking[:10]})

@app.route('/search')
def search():
    query = request.args.get('query', '').lower()
    if not query:
        return jsonify([])

    # Create query vector
    query_vector = np.zeros(len(vocab))
    query_terms = query.split()

    contains_booleans = False

    # For creating query vector, modified to also find suggested words
    typo_suggestions = []
    for term in query_terms:
        if term in vocab:
            idx = vocab.index(term)
            query_vector[idx] += 1
        elif term == "or" or term == "and":
            contains_booleans = True
        else:
            typo_suggestions += find_closest(term, vocab)
    print(typo_suggestions)

    query_vocab_terms = [term for term in query.replace("or", "").replace("and", "").split() if term in vocab]
    if not query_vocab_terms: 
        return jsonify({
            "results": [],
            "suggestions": typo_suggestions
        })

    # Create document-term matrix
    doc_term_matrix = create_doc_term(complex_items, vocab, mode="tf")
    doc_term_binary = np.where(doc_term_matrix > 0, 1, 0)

    query_vector = construct_query_vec(query_vocab_terms)
    query_vector_bin = np.where(query_vector > 0, 1, 0)

    if contains_booleans: 
        if " or " in query: 
            bool_mask = boolean_or(query_vector_bin, doc_term_binary)
        else: 
            bool_mask = boolean_and(query_vector_bin, doc_term_binary)
    else: 
        bool_mask = np.ones(len(complexRep), dtype=bool) #doesn't do anything

    filtered_matrix = doc_term_tf_rep[bool_mask]
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

            if title in already_seen: 
                continue
            else: 
                already_seen.add(title)

            results.append({
                'title': title,
                'similarity': float(score)
            })

    # Sort by similarity score descending
    results.sort(key=lambda x: x['similarity'], reverse=True)

    for item in results[:10]:
        print(repr(item))

    print("hm", typo_suggestions)
    return jsonify({
        "results": results[:10], 
        "suggestions": typo_suggestions 
    })  # Return top 10 results


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)