#run.py

import sys
sys.path.append('D:/Varsity/Honours/Semester 2/Comp700/SpotifyMusicRecommendationSystem/SpotifyMusicRecommendationSystem')

from app import create_app
from app.main import initialize_clustering

app = create_app()

if __name__ == "__main__":
    # Initialize clustering
    with app.app_context():
        initialize_clustering()
    
    # Start the Flask development server
    app.run(debug=True)

