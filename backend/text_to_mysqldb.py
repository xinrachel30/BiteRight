import os

try:
    with open("tempDataset.txt", "r") as file:
        content = file.read()
        print("Read file successfully")
        
        dictionary = eval(content)

        if not dictionary:
            print("Dictionary is empty!")
        dictionary = {key.replace("\n", " ").replace('"', "").replace("'", ""): value for key, value in dictionary.items()}
        
    output_file = "food.sql"
    with open(output_file, "w") as output:
        output.write("DROP TABLE IF EXISTS fooddb;\n")
        output.write("\n")
        output.write("CREATE TABLE fooddb(\n")
        output.write("    comment varchar(1000),\n")
        output.write("    upvotes int\n")
        output.write(");\n\n")
        
        for key, value in dictionary.items():
            sql = f"INSERT INTO fooddb (comment, upvotes) VALUES ('{key}', {value});\n"
            output.write(sql)

except FileNotFoundError:
    print("Couldn't find the file")
except Exception as e:
    print(f"Other error: {e}")
