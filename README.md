This research is a comparison of 3 clustering algorithms - K-means, DBSCAN and Agglomerative. <br/>
We are comparing these algorithms in regards to the clustering of music and then comparing which algorithm recommending the best music. <br/>
The dataset used in this research is from Spotify. It consists of over 1 million songs from 2000-2022 with numerous details about each song(song name, artist, tempo, energy, popularity, valence, etc). <br/>
This is a Flask web application done primarily in Python. HTML and CSS was used for the aesthetics. HTML and Javascript was used to make the program interactive. Numerous libraries such as numpy, pandas, sqlalchmey and scikit-learn was used for the functionality of the system<br/>

In order to run the system:

Go to your command prompt.
Change directory to where the project is located.
Type "python run.py" and press enter
The system should be running as localhost on your default web browser
You can copy the web address from the terminal in your IDE
Below is a list of all the pages in the system and what they do:

/app : app folder with system files inside
/clustering : Folder with clustering files
init.py : User to initialize a python package
kmeans.py : file with the required code for kmeans clustering
dbscan.py : file with the required code for dbscan clustering
agglomerative.py : file with the required code for agglomerative clustering
/static : Folder with CSS files and images
/css : Folder with CSS files
app.css : css for main file
comparison.css : css for comparison page(page used to compare the algorithms)
results/css : css for results page(page where recommendations are displayed)
/graphs : Folder with graphs used to compare the clustering algorithms
images : Folder with images used in system
/templates : Folder with HTML files
app.html : HTML for main page
comparison.html : HTML for page used to compare algorithms
results.html : HTML for page with recommendations from each algorithm
search.html : HTML for search table of app.html
init.py : Used to initialize a Python package
main.py : Entry point of the system. Used as the main file and to direct to other files for the required functionality
models.py : Database models are defined here
config.py : Used for security
dataset.cv : Dataset used in the research. This is for downloading and viewing purposes as the dataset was copied onto a SQL Server Management Studio as a database.
requirements.txt : Shows that the system is a flask application
run.py : file used to run the system
