# BiteRight

To clone this repository, run `git clone git@github.com:xinrachel30/BiteRight.git <name_of_directory>`
To enter the project directory, run `cd <name_of_directory>`

### Setting up MySQL
You will need to install MySQL. Here are some tutorials to help with this:
- Windows: https://blog.devart.com/how-to-install-mysql-on-windows-using-mysql-installer.html
  - Select CUSTOM installation and remove any Visual Studio dependencies
- Mac: Preferably use homebrew. Your default password will be empty (""). If not, follow this https://www.geeksforgeeks.org/how-to-install-mysql-on-macos/
- Linux: https://www.geeksforgeeks.org/how-to-install-mysql-on-linux/

### Create and activate a virtual environment in Python

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
mysql -u <username> -p biterightdb < backend/food.sql
```
Then, in backend/app.py, edit the SQL credentials at the top of the file to match your local MySQL credentials (such as variables LOCAL_MYSQL_USER and LOCAL_MYSQL_USER_PASSWORD) 

You can enter the backend folder by running `cd backend`
Finally, from the backend folder, run the following to begin the application.

```flask run --host=0.0.0.0 --port=5000```
