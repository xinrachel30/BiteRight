import json
import os
import re
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from python_scripts_and_data.jaccard_sim import *
from python_scripts_and_data.jaccard_sim import create_doc_term
from python_scripts_and_data.unsupervised import *
from python_scripts_and_data.evaluation import *
from python_scripts_and_data.cos_sim import *
import numpy as np
import colorama
from colorama import Fore, Style

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
#LOCAL_MYSQL_USER = "wsl_root"
LOCAL_MYSQL_USER = "root"
LOCAL_MYSQL_USER_PASSWORD = "qwertyui" # TODO: make this an env variable
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

@app.route("/vibe-search")
def vibe_search():
    query_foods = request.args.get('food_vibe', '').lower() 
    if not query_foods:
        return jsonify([])
    
    tokenized_query = tokenize_query(query_foods.replace(',', '')) 
    comma_separated = [section.strip() for section in query_foods.split(',') if section.strip()]

    cleaned_query_boolean = []
    for part in comma_separated: 
        tokens = tokenize_query(part)
        parsed = parse_parens(tokens)
        #print("each part: ", part, "->", parsed)
        cleaned_query_boolean.append(parsed)

    final_boolean = []
    for i, parsed in enumerate(cleaned_query_boolean):
        if i > 0:
            final_boolean.append("or") # adds 'or' between every two "parts"
        final_boolean.append(parsed) 

    query_vector = np.zeros(len(vocab))
    typo_suggestions = []
    contains_booleans = False

    # For creating query vector, modified to also find suggested words
    final_query = []
    for term in tokenized_query:
        if term in vocab:
            idx = vocab.index(term)
            query_vector[idx] += 1
            final_query.append(term)
            #print(term, "is a vocab word")
        elif (not contains_booleans) and term in ["or", "and", "not", "(", ")"]: #considering parentheses as booleans
            contains_booleans = True
            #print("contains booleans")
        else:
            #print("needed to find a suggestion for", term)
            typo_suggestions.extend(find_closest(term, vocab)) #if misspelt, find typos suggestions
    

    results = closest_food_profile(final_query)
    #dict of {food: score}
    top_10 = list(results.keys())[:10]

    results_list = []
    for food in top_10:
        if results[food] > 0:
            results_list.append({
                'food': food,
                'similarity': results[food]
            })

    return jsonify({
        "results": top_10, 
        "suggestions": typo_suggestions,
    })  # Return top 10 results


@app.route('/search')
def search():
    query = request.args.get('food', '').lower()    
    selected_flavors = request.args.get('flavors', '').lower()
    
    if not query and not selected_flavors:
        return jsonify([])

    selected_flavors = re.split(r'[,\s]+', selected_flavors.strip())
    print("\n\n\n\n", selected_flavors)
    flavors_wanted = set(selected_flavors)
    unspecified_flavors = [flavor for flavor in selected_flavors if flavor not in flavors]
    print("i need to find some suggestions for ", unspecified_flavors)

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
    
    flavor_typo_suggestions = []
    for flavor in unspecified_flavors: 
        if flavor == "": 
            continue
        flavor_typo_suggestions.extend(find_closest(flavor, flavors))

    # print(flavors)
    print("the flavor typos suggestions i found wereee", flavor_typo_suggestions)
    query_vocab_terms = [term for term in tokenized_query if term in vocab]
    # flavors_have = closest_food_profile(query_vocab_terms)

    print("final list of vocab in query: ", query_vocab_terms)
    
    print("typo suggestions: ", typo_suggestions)
    if not query_vocab_terms: #no vocab words 
        return jsonify({
            "results": [],
            "suggestions": typo_suggestions, 
            "flavor_suggestions": flavor_typo_suggestions
        })

    # Create document-term matrix
    doc_term_matrix = create_doc_term(complex_items, vocab, mode="tf")
    # print("doc term matrix shape: ", doc_term_matrix.shape)
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

    # Get Jaccard scores
    jaccard_scores = gen_jaccard_sim(query_vector, filtered_matrix)
    
    cosine_scores = main_cos(query_vector, filtered_matrix, indices)
    cosine_scores = cosine_scores[indices]

    jaccard_weight = 0.6 # change this value to test best combination
    combined = jaccard_weight * jaccard_scores + (1 - jaccard_weight) * cosine_scores

    # Get top results where similarity > 0
    results = []
    already_seen = set() #remove duplicates

    sorted_query_vocab = tuple(sorted(query_vocab_terms))

    for idx, score in enumerate(combined):
        if score > 0:
            true_idx = indices[idx]
            comment_id, (food_dict, _) = complex_items[true_idx]
            food_items = list(food_dict.keys())

            sorted_food_items = sorted(food_items)
            title = ", ".join(sorted_food_items).strip()
            key = tuple(sorted_food_items)

            if key in already_seen or len(food_items) == 0 or key == sorted_query_vocab: 
                continue
            if jaccard_scores[idx] == 0: 
                continue
            if len(food_items) > 6: 
                continue
            already_seen.add(key)

            result_flavor_dict = closest_flavors_given_foods(food_items)
            result_flavor_set = set(f.lower() for f in result_flavor_dict.keys())

            flavor_weight = 0.5
            combined_score = combined[idx] * (1-flavor_weight)
            flavor_score = 0
            if flavors_wanted:
                print(Fore.BLUE + "User wants flavors:", flavors_wanted)
                print("Predicted flavors:", result_flavor_set)

                intersection = flavors_wanted & result_flavor_set
                numerator = sum(result_flavor_dict[item] for item in intersection)
                denominator = len(intersection)

                if denominator > 0:
                    flavor_score = numerator / denominator
                    combined_score = flavor_score * flavor_weight + combined_score * (1 - flavor_weight)

            # format flavor desc
            top_flavors = list(result_flavor_dict.keys())[:3]
            flavor_desc = ", ".join(top_flavors[:-1]) + ", and " + top_flavors[-1] if len(top_flavors) >= 3 else ", ".join(top_flavors)

            # format scores
            f_jaccard = "{:.2f}".format(jaccard_scores[idx])
            f_cosine = "{:.2f}".format(cosine_scores[idx])
            f_flavor = "{:.2f}".format(flavor_score * 100)
            f_combined = "{:.2f}".format(combined_score * 100)
            f_without_flavor = "{:.2f}".format(combined[idx] * 100)

            score_txt = f"Trends Match: {f_without_flavor}% (jaccard: {f_jaccard}, cosine: {f_cosine})"
            if flavors_wanted:
                score_txt += f", Flavor Match: {f_flavor}%"

            results.append({
                'title': title,
                'similarity': score_txt,
                'combined_score': combined_score,
                'flavor_desc': flavor_desc
            })

    results.sort(key=lambda x: x['combined_score'], reverse=True)
    top_10 = results[:10]

    for result in top_10:
        result.pop('combined_score', None)
        print(Fore.BLUE + "User wants flavors:", flavors_wanted)
        print("Predicted flavors:", result["flavor_desc"])
        print(Style.RESET_ALL)


    return jsonify({
        "results": top_10, 
        "suggestions": typo_suggestions,
        "flavor_suggestions": flavor_typo_suggestions
    })  # Return top 10 results


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
    