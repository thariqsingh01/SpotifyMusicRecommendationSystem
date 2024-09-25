#run.py

"""
import os
import sys
sys.path.append('D:/Varsity/Honours/Semester 2/Comp700/SpotifyMusicRecommendationSystem/SpotifyMusicRecommendationSystem')

os.environ["OMP_NUM_THREADS"] = "1"

from app import create_app
from app.main import initialize_clustering

app = create_app()

if __name__ == "__main__":
    # Initialize clustering
    with app.app_context():
        initialize_clustering()
    
    # Start the Flask development server
    app.run(debug=True)

"""

import os
import sys
import logging
import urllib
from app import create_app
from app.main import initialize_clustering

sys.path.append('D:/Varsity/Honours/Semester 2/Comp700/SpotifyMusicRecommendationSystem/SpotifyMusicRecommendationSystem')

os.environ["OMP_NUM_THREADS"] = "1"


if __name__ == "__main__":
    app, engine = create_app('mssql+pyodbc:///?odbc_connect=' + urllib.parse.quote_plus(
            'DRIVER={ODBC Driver 17 for SQL Server};SERVER=MSCS\\SQLEXPRESS;DATABASE=Spotify;Trusted_Connection=yes;'
        ))  
    with app.app_context():  
        initialize_clustering('mssql+pyodbc:///?odbc_connect=' + urllib.parse.quote_plus(
            'DRIVER={ODBC Driver 17 for SQL Server};SERVER=MSCS\\SQLEXPRESS;DATABASE=Spotify;Trusted_Connection=yes;'
        ), engine) 
    app.run()



