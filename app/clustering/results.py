# results.py

from app import db
from app.models import SpotifyData
from sqlalchemy import func
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Set up Spotipy with your credentials
client_credentials_manager = SpotifyClientCredentials(client_id='YOUR_CLIENT_ID', client_secret='YOUR_CLIENT_SECRET')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_user_choices(user_id):
    # Retrieve user choices from the database based on user_id
    return db.session.query(SpotifyData).filter(SpotifyData.user_id == user_id).all()

def get_song_details(track_id):
    """Fetch song details using Spotipy."""
    try:
        track = sp.track(track_id)
        return {
            'track_id': track['id'],
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'album_cover': track['album']['images'][0]['url'] if track['album']['images'] else None
        }
    except Exception as e:
        print(f"Error fetching track details for {track_id}: {e}")
        return None

def generate_recommendations(user_choices):
    recommendations = {
        'KMeans': [],
        'Agglomerative': [],
        'DBSCAN': []
    }

    # For each user choice, find top 5 recommendations based on clustering results
    for choice in user_choices:
        # Get top 5 recommendations for KMeans
        kmeans_recommendations = (
            db.session.query(SpotifyData)
            .filter(SpotifyData.kmeans == choice.kmeans)
            .order_by(SpotifyData.popularity.desc())
            .limit(5)
            .all()
        )
        for song in kmeans_recommendations:
            details = get_song_details(song.track_id)
            if details:
                recommendations['KMeans'].append(details)

        # Get top 5 recommendations for Agglomerative
        agglomerative_recommendations = (
            db.session.query(SpotifyData)
            .filter(SpotifyData.agglomerative == choice.agglomerative)
            .order_by(SpotifyData.popularity.desc())
            .limit(5)
            .all()
        )
        for song in agglomerative_recommendations:
            details = get_song_details(song.track_id)
            if details:
                recommendations['Agglomerative'].append(details)

        # Get top 5 recommendations for DBSCAN
        dbscan_recommendations = (
            db.session.query(SpotifyData)
            .filter(SpotifyData.dbscan == choice.dbscan)
            .order_by(SpotifyData.popularity.desc())
            .limit(5)
            .all()
        )
        for song in dbscan_recommendations:
            details = get_song_details(song.track_id)
            if details:
                recommendations['DBSCAN'].append(details)

    return recommendations
