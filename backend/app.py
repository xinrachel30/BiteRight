import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from python_scripts_and_data.jaccard_sim import create_doc_term, gen_jaccard_sim, vocab, complexRep
import numpy as np

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
LOCAL_MYSQL_USER = "root"
LOCAL_MYSQL_USER_PASSWORD = "3v3ryJ0b@pplic" # TODO: make this an env variable
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

@app.route('/search')
def search():
    query = request.args.get('query', '').lower()
    if not query:
        return jsonify([])

    # Create query vector
    query_vector = np.zeros(len(vocab))
    query_terms = query.split()
    
    for term in query_terms:
        if term in vocab:
            idx = vocab.index(term)
            query_vector[idx] += 1

    # Create document-term matrix
    doc_term_matrix = create_doc_term(complexRep, vocab, mode="tf")
    
    # Get similarity scores using generalized Jaccard
    similarity_scores = gen_jaccard_sim(query_vector, doc_term_matrix)
    
    # Get top results where similarity > 0
    results = []
    for idx, score in enumerate(similarity_scores):
        if score > 0:
            food_items = list(complexRep[idx][0].keys())
            results.append({
                'title': ', '.join(food_items),
                'similarity': float(score)
            })
    
    # Sort by similarity score descending
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return jsonify(results[:10])  # Return top 10 results


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)