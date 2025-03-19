import os

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
        words = words_dict.keys()

        for comment_key, word_key in zip(comments, words):
            upvotes = comment_dict.get(comment_key)
            sql = f"INSERT INTO fooddb (upvotes, words, comment) VALUES ('{upvotes}','{word_key}', '{comment_key}');\n"
            output.write(sql)

except FileNotFoundError as e:
    print(f"Couldn't find the file: {e}")
except Exception as e:
    print(f"Other error: {e}")