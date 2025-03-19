import os
import re

try:
    with open("python_scripts_and_data/data/comment_score_dict.txt", "r") as file:
        content = file.read()
        print("Read comment file successfully")
        comment_dict = eval(content)

        if not comment_dict:
            print("Comment dictionary is empty!")
        comment_dict = {key.replace("\n", " ").replace('"', "").replace("'", ""): value for key, value in comment_dict.items()}
    
    with open("python_scripts_and_data/data/foodwords_score_dict.txt", "r") as file:
        content = file.read()
        print("Read words file successfully")
        words_dict = eval(content)

        if not words_dict:
            print("Food words dictionary is empty!")
        # each item in foodwords_score_dict should already be pre-processed -> omit replacing newlines quotation marks
    
    vocab = []
    with open(os.path.join("python_scripts_and_data/data", "foodVocab.txt"), "r") as f: 
        raw_vocab = f.read().lower()
        vocab = raw_vocab.split(", ")
    vocab = set(vocab)

    output_file = "food.sql"
    with open(output_file, "w") as output:
        output.write("DROP TABLE IF EXISTS fooddb;\n")
        output.write("\n")
        output.write("CREATE TABLE fooddb(\n")
        output.write("    comment_id SERIAL PRIMARY KEY,\n")
        output.write("    upvotes INT,\n")
        output.write("    words TEXT,\n")
        output.write("    comment MEDIUMTEXT\n")
        output.write(");\n\n")
        
        comments = comment_dict.keys()
        hungry = 5

        for comment in comments:
            # <pre-processing>
            words = re.sub(r"[^\w\s]", "", comment)
            comment_word_list = words.lower().split()
            
            food_words = ""
            for key_word in vocab: 
                if key_word in comment_word_list :
                    food_words += str(key_word + ", ")
            if food_words != "":
                food_words = food_words[:-2]


            if hungry > 0: 
                print(food_words)
                hungry -= 1
            # </pre-processing>

            upvotes = comment_dict.get(comment)
            sql = f"INSERT INTO fooddb (upvotes, words, comment) VALUES ('{upvotes}','{food_words}', '{comment}');\n"
            output.write(sql)

except FileNotFoundError as e:
    print(f"Couldn't find the file: {e}")
except Exception as e:
    print(f"Other error: {e}")