#run.py

import sys
sys.path.append('D:/Varsity/Honours/Semester 2/Comp700/SpotifyMusicRecommendationSystem/SpotifyMusicRecommendationSystem')

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
