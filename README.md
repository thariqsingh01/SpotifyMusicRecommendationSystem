This research is a comparison of 3 clustering algorithms - K-means, DBSCAN and Agglomerative. <br/>
We are comparing these algorithms in regards to the clustering of music and then comparing which algorithm recommending the best music. <br/>
The dataset used in this research is from Spotify. It consists of over 1 million songs from 2000-2022 with numerous details about each song(song name, artist, tempo, energy, popularity, valence, etc). <br/>
This is a Flask web application done primarily in Python. HTML and CSS was used for the aesthetics. HTML and Javascript was used to make the program interactive. Numerous libraries such as numpy, pandas, sqlalchmey and scikit-learn was used for the functionality of the system<br/>

In order to run the system:

Go to your command prompt.<br/>
Change directory to where the project is located.<br/>
Type "python run.py" and press enter.<br/>
The system should be running as localhost on your default web browser.<br/>
You can copy the web address from the terminal in your IDE.<br/>

Below is a list of all the pages in the system and what they do:<br/>

/app : app folder with system files inside<br/>
&emsp;/clustering : Folder with clustering files<br/>
&emsp;&emsp;init.py : User to initialize a python package<br/>
&emsp;&emsp;kmeans.py : file with the required code for kmeans clustering<br/>
&emsp;&emsp;dbscan.py : file with the required code for dbscan clustering<br/>
&emsp;&emsp;agglomerative.py : file with the required code for agglomerative clustering<br/>
&emsp;/static : Folder with CSS files and images<br/>
&emsp;&emsp;/css : Folder with CSS files<br/>
&emsp;&emsp;&emsp;app.css : css for main file<br/>
&emsp;&emsp;&emsp;comparison.css : css for comparison page(page used to compare the algorithms)<br/>
&emsp;&emsp;&emsp;results/css : css for results page(page where recommendations are displayed)<br/>
&emsp;&emsp;/graphs : Folder with graphs used to compare the clustering algorithms<br/>
&emsp;&emsp;/images : Folder with images used in system<br/>
&emsp;/templates : Folder with HTML files<br/>
&emsp;&emsp;app.html : HTML for main page<br/>
&emsp;&emsp;comparison.html : HTML for page used to compare algorithms<br/>
&emsp;&emsp;results.html : HTML for page with recommendations from each algorithm<br/>
&emsp;&emsp;search.html : HTML for search table of app.html<br/>
init.py : Used to initialize a Python package<br/>
main.py : Entry point of the system. Used as the main file and to direct to other files for the required functionality<br/>
models.py : Database models are defined here<br/>
config.py : Used for security<br/>
dataset.cv : Dataset used in the research. This is for downloading and viewing purposes as the dataset was copied onto a SQL Server Management Studio as a database.<br/>
requirements.txt : Shows that the system is a flask application<br/>
run.py : file used to run the system<br/>
