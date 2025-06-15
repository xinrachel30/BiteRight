# BiteRight

## Project Description

BiteRight is a food pairing recommendation system. We have two search options: ingredient search and vibe search. <ins>Ingredient search</ins> allows users to enter a list of foods they are eating (ie. banana and bread) and some flavors they'd like to incorporate into the dish (ie. sweet) to get recommendations on what foods they can additionally incorporate into the meal. In each result are some metrics conveying why some results are ranked more favorably than others. Such metrics include jaccard and cosine scores, trend match, and flavor match. Trend match describes how closely the recommendation matches the opinions and insights of recent comments made by Reddit users. Flavor match describes how closely the dish represents the flavor that was inputted as a query. 

On the other side of the screen is <ins>vibe search</ins>, which allows users to enter some combination of foods and receive foods which have a similar vibe. For example, if the query is "peach apple banana," the top result is "butternut squash." This result aligns with peaches and apples because it is sugary and fruity. It additionally matches with banana because it is tender. These latent dimensions are displayed to users as well. 

### Technologies Used

Our program uses HTML, CSS, and vanilla JavaScript for the frontend components. We used Python, Flask, and MySQL for the backend. Other core algorithms and technologies include: 

- Binary vector representation, TF-IDF representation, and Term-Document Matrix: Used in our implementation of cosine similarity and in processing and storing scraped data
- Cosine similarity: Used in combination with Jaccard to find similarity scores between a query and a document
- Jaccard similarity: Used in combination with Cosine to find similarity scores between a query and a document
- Edit distance: Used for typo correction to compare non-food words found in the query with our database of food words. We use edit distance to find the closest two matches and give suggestions to the users. We implemented a similar algorithm to provide suggestions for flavors as well.
- SVD: Used to quantify flavor profiles of a combination of foods for both Ingredient Search and Vibe Search. Originally (deprecated after P04, but still implemented), we also used SVD to search foods along multiple selectable flavor dimensions.

## Instructions to Install and Run

To clone this repository, run `git clone git@github.com:xinrachel30/BiteRight.git <name_of_directory>`
To enter the project directory, run `cd <name_of_directory>`

### Step 1: Setting up MySQL
You will need to install MySQL. Here are some tutorials to help with this:
- Windows: https://blog.devart.com/how-to-install-mysql-on-windows-using-mysql-installer.html
  - Select CUSTOM installation and remove any Visual Studio dependencies
- Mac: Preferably use homebrew. Your default password will be empty (""). If not, follow this https://www.geeksforgeeks.org/how-to-install-mysql-on-macos/
- Linux: https://www.geeksforgeeks.org/how-to-install-mysql-on-linux/

### Step 2: Create and activate a virtual environment in Python

Run `python3 -m venv <virtual_env_name>` in the project directory to create a new virtual environment.

To activate: 
- Windows: <virtual_env_name>\Scripts\activate
- Mac/Linux: source <virtual_env_name>/bin/activate

### Step 3: Install dependencies

From the project directory, you can run `python3 -m pip install -r backend/requirements.txt`

### Step 4: Connection to MySQL

Make sure the MySQL server is running. Enter the MySQL prompt using `mysql -u <username> -p` and enter your password when prompted. Then, follow the next steps to create and populate the database. 

```
CREATE DATABASE biterightdb;
USE biterightdb;
quit;
mysql -u <username> -p biterightdb < food-dump.sql
```
Then, in backend/app.py, edit the SQL credentials at the top of the file to match your local MySQL credentials (such as variables LOCAL_MYSQL_USER and LOCAL_MYSQL_USER_PASSWORD) 

You can enter the backend folder by running `cd backend`
Finally, from the backend folder, run the following to begin the application.

```flask run --host=0.0.0.0 --port=5000```
