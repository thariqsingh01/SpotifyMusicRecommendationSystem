import os
import sys
import logging
import urllib
from app import create_app
from app.main import initialize_clustering  # Import clustering function

sys.path.append('D:/Varsity/Honours/Semester 2/Comp700/SpotifyMusicRecommendationSystem/SpotifyMusicRecommendationSystem')

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    # Create the connection string
    connection_string = 'mssql+pyodbc:///?odbc_connect=' + urllib.parse.quote_plus(
        'DRIVER={ODBC Driver 17 for SQL Server};SERVER=MSCS\\SQLEXPRESS;DATABASE=Spotify;Trusted_Connection=yes;')

    # Initialize app and engine
    app, engine = create_app(connection_string)  
    
    with app.app_context():  
        # Initialize clustering with connection_string and engine
        initialize_clustering(connection_string, engine) 
    
    app.run(debug=True)
