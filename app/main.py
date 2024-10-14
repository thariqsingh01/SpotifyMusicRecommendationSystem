#main.py

from flask import Flask, Blueprint, render_template, request, current_app, jsonify
from .models import SpotifyData
from app import db
from .clustering.kmeans import perform_kmeans_clustering, generate_kmeans_graph
from .clustering.dbscan import perform_dbscan_clustering, generate_dbscan_graph
from .clustering.agglomerative import perform_agglomerative_clustering, generate_agglomerative_graph
from .clustering.comparison import generate_cnn_graph  # Import CNN graph generation
import pandas as pd
import logging
from flask_caching import Cache
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
load_dotenv()

# Set up Spotipy with your Spotify API credentials
client_id = os.getenv('SPOTIFY_CLIENT_ID', 'default_client_id')
client_secret = os.getenv('SPOTIFY_CLIENT_SECRET', 'default_client_secret')
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

bp = Blueprint('main', __name__)

from sqlalchemy import or_

def get_user_choices(request_data):
    """
    Extracts user-selected songs from the submitted JSON data.
    Args:    request_data (dict): A list of dictionaries containing song data submitted by the user.
    Returns: list: A list of SpotifyData objects representing user-selected songs.
    """

    # Extract track_name and artist_name pairs from request_data
    track_artist_pairs = [(item['track_name'], item['artist_name']) 
                          for item in request_data 
                          if item.get('track_name') and item.get('artist_name')]
    
    if not track_artist_pairs:
        return []
    
    # Use 'or_' to filter by multiple conditions (track_name and artist_name)
    user_choices = SpotifyData.query.filter(
        or_(
            *[(SpotifyData.track_name == track_name) & (SpotifyData.artist_name == artist_name) 
              for track_name, artist_name in track_artist_pairs]
        )
    ).all()

    for song in user_choices:
        logger.info(f'This is the current song: {song}')
    
    return user_choices


def calculate_similarity(user_features, song_features):
    """
    Calculates the cosine similarity between the user's selected song features and a given song's features.
    Args:    user_features (list): A list of features from the user's selected song (e.g., danceability, energy).
             song_features (list): A list of features from a Spotify song.
    Returns: float: The cosine similarity score between the user's song and the given song.
    """
    return cosine_similarity([user_features], [song_features])[0][0]

def get_track_cover(sp, track_id):
    """
    Fetches the album cover URL for a given track using the Spotify API.
    Args:     sp (Spotipy object): The Spotipy client instance.
              track_id (str): The Spotify track ID.
    Returns:  str: URL of the album cover image, or None if unavailable.
    """
    try:
        track = sp.track(track_id)
        return track['album']['images'][0]['url'] if track['album']['images'] else None
    except Exception as e:
        logger.error(f"Error fetching cover for track_id {track_id}: {str(e)}")
        return None

def fetch_covers_for_user_choices(user_choices):
    """
    Fetches album cover URLs for the list of user-selected songs.
    Args:    user_choices (list): A list of SpotifyData objects representing user-selected songs.
    Returns: list: A list of album cover URLs.
    """
    track_ids = [song.track_id for song in user_choices]
    
    # Fetch covers for all user songs
    covers = []
    for track_id in track_ids:
        cover_url = get_track_cover(sp, track_id)
        if cover_url:
            covers.append(cover_url)
    
    return covers


# Modify the generate_recommendations function to include cover images
def generate_recommendations(user_song, cluster_label_field, limit=10):
    """
    Generates recommendations for the user based on their selected song and assigned cluster label.
    Args:
        user_song (SpotifyData object): The song chosen by the user.
        cluster_label_field (str): The name of the cluster label field (e.g., 'kmeans', 'dbscan', 'agglomerative').
        limit (int): The maximum number of songs to consider for recommendations.
    Returns:
        list: A list of recommended songs with similarity scores and album covers.
    """
    # Log starting point
    logger.info(f"Starting recommendation generation for user song: {user_song.track_name} by {user_song.artist_name}, "
                f"KMeans label: {user_song.kmeans}")

    # Query for songs in the same cluster as the user's song, limiting the results
    similar_songs = SpotifyData.query.filter(
        getattr(SpotifyData, cluster_label_field) == getattr(user_song, cluster_label_field)
    ).limit(limit).all()

    # Log the number of similar songs found
    logger.info(f"Number of similar songs found for {cluster_label_field}: {len(similar_songs)}")
    logger.info(f"Cluster label used for filtering: {getattr(user_song, cluster_label_field)}")

    # Extract relevant features from the user's selected song
    user_features = [user_song.danceability, user_song.energy, user_song.acousticness, user_song.valence]
    logger.info(f"User features: {user_features}")

    recommendations = []
    for song in similar_songs:
        # Log each song's basic details
        logger.info(f"Song in cluster: {song.track_name} by {song.artist_name}, Features: "
                    f"[{song.danceability}, {song.energy}, {song.acousticness}, {song.valence}]")

        # Extract the features of each song in the same cluster
        song_features = [song.danceability, song.energy, song.acousticness, song.valence]

        # Calculate similarity score between the user's song and the current song
        similarity_score = calculate_similarity(user_features, song_features)
        logger.info(f"Calculated similarity score for song {song.track_name}: {similarity_score}")

        # Append the song and similarity score to the recommendations list
        recommendations.append((song, similarity_score))

    # Sort recommendations by similarity score in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Fetch covers only for the top 5 recommendations
    top_recommendations = []
    for song, similarity_score in recommendations[:5]:
        # Fetch the cover image using Spotipy
        cover_url = get_track_cover(sp, song.track_id)  # Ensure individual track_id is passed here
        logger.info(f"Fetching cover image for song {song.track_name}...")

        if cover_url:  # Ensure cover_url is not None before proceeding
            # Append the song, its similarity score, and cover URL to the top recommendations
            top_recommendations.append({
                'track_id': song.track_id,
                'artist_name': song.artist_name,
                'track_name': song.track_name,
                'year': song.year,
                'genre': song.genre,
                'cover_url': cover_url  # Add cover_url to the recommendation
            })

    logger.info(f"Top 5 recommendations: {[rec['track_name'] for rec in top_recommendations]}")
    return top_recommendations

@bp.route('/search', methods=['GET'])
def search():
    query = request.args.get('q')

    results = []
    if query:
        results = SpotifyData.query.filter(
            (SpotifyData.artist_name.ilike(f"%{query}%")) |
            (SpotifyData.track_name.ilike(f"%{query}%")) |
            (SpotifyData.year.ilike(f"%{query}%")) |
            (SpotifyData.genre.ilike(f"%{query}%"))
        ).order_by(SpotifyData.popularity.desc()).limit(20).all()
        logger.info(f'Number of results found: {len(results)}')  # Log number of results

    if not results:
        logger.info('No results matched the query.')

    # Render only the table rows for the HTMX request
    return render_template('search.html', results=results)

@bp.route('/recommendations', methods=['POST'])
def recommendations():
    try:
        logger.info(f'request.json: {request.json}')
        user_choices = get_user_choices(request.json)

        if not user_choices:
            return jsonify({"message": "No choices provided"}), 400

        selected_song = user_choices[0]
        selected_song_name = selected_song.track_name
        selected_artist = selected_song.artist_name

        logger.info(f'User choices: {user_choices}')
        logger.info(f'Selected song details: Track ID: {selected_song.track_id}, Track Name: {selected_song.track_name}, Artist: {selected_song.artist_name}')


        user_song = selected_song

        if not user_song:
            return jsonify({"message": "Song not found"}), 404

        # Log the cluster labels for debugging
        logger.info(f'Track ID: {user_song.track_id}, Track Name: {user_song.track_name}, Artist: {user_song.artist_name}')
        logger.info(f'KMeans label: {user_song.kmeans}, DBSCAN label: {user_song.dbscan}, Agglomerative label: {user_song.agglomerative}')

        # Retrieve cluster labels for the selected song
        kmeans_cluster_label = user_song.kmeans  # KMeans label
        dbscan_cluster_label = user_song.dbscan  # DBSCAN label
        agglomerative_cluster_label = user_song.agglomerative  # Agglomerative label

        # Validate the cluster labels
        if kmeans_cluster_label is None or dbscan_cluster_label is None or agglomerative_cluster_label is None:
            return jsonify({"message": "Cluster labels not found"}), 404

        # Generate recommendations for each clustering algorithm
        kmeans_recommendations = generate_recommendations(user_song, 'kmeans', limit=20)
        dbscan_recommendations = generate_recommendations(user_song, 'dbscan', limit=20)
        agglomerative_recommendations = generate_recommendations(user_song, 'agglomerative', limit=20)

        # Check if any recommendations are returned
        if not kmeans_recommendations and not dbscan_recommendations and not agglomerative_recommendations:
            logger.warning("No recommendations found for any of the clustering methods")
            return render_template('results.html', 
                                   message="No recommendations found for the selected song.")

        return render_template('results.html',
                               kmeans_recommendations=kmeans_recommendations,
                               dbscan_recommendations=dbscan_recommendations,
                               agglomerative_recommendations=agglomerative_recommendations)

    except Exception as e:
        logger.error(f'Error in recommendations: {e}', exc_info=True)
        return jsonify({"message": "An error occurred"}), 500


@bp.route('/suggestions', methods=['GET'])
def suggestions():
    track_id = request.args.get('track_id')
    logger.info(f'Suggestions request for track_id: {track_id}')  # Log the request

    if not track_id:
        logger.error('Track ID is required for suggestions.')
        return {'error': 'Track ID is required.'}, 400

    # Find the KMeans cluster for the selected track (modify for chosen algorithm)
    selected_track = SpotifyData.query.filter_by(track_id=track_id).first()
    if not selected_track or not selected_track.kmeans:  # Adjust for chosen algorithm
        logger.error('Track not found or not clustered.')
        return {'error': 'Track not found or not clustered.'}, 404

    cluster = selected_track.kmeans

    # Fetch other songs in the same cluster
    similar_songs = SpotifyData.query.filter_by(kmeans=cluster).limit(10).all()
    logger.info(f"Number of similar songs found: {len(similar_songs)}")
    for song in similar_songs:
        logger.info(f"Considering song: {song.track_name} by {song.artist_name}")


    # Prepare suggestions for rendering
    suggestions_list = [
        {
            'track_id': song.track_id,
            'artist_name': song.artist_name,
            'track_name': song.track_name,
            'year': song.year,
            'genre': song.genre,
            'duration': song.duration_in_minutes_seconds(),  # Assuming this method exists
        }
        for song in similar_songs
    ]

    logger.info(f'Number of suggestions found: {len(suggestions_list)}')  # Log the number of suggestions
    return jsonify({'suggestions': suggestions_list}), 200

@bp.route('/')
def home():
    return render_template("app.html")

@bp.route('/comparison', methods=['GET', 'POST'])
@cache.cached(timeout=300)
def comparison():
    with current_app.app_context():
        # Get the database engine
        engine = db.engine

        # Single query to retrieve cluster groups and their counts for KMeans, DBSCAN, and Agglomerative
        cluster_query = """
            SELECT
                kmeans AS cluster_group,
                COUNT(*) AS count,
                'KMeans' AS algorithm
            FROM Spotify
            WHERE kmeans IS NOT NULL
            GROUP BY kmeans

            UNION ALL

            SELECT
                dbscan AS cluster_group,
                COUNT(*) AS count,
                'DBSCAN' AS algorithm
            FROM Spotify
            WHERE dbscan IS NOT NULL
            GROUP BY dbscan

            UNION ALL

            SELECT
                agglomerative AS cluster_group,
                COUNT(*) AS count,
                'Agglomerative' AS algorithm
            FROM Spotify
            WHERE agglomerative IS NOT NULL
            GROUP BY agglomerative

            ORDER BY algorithm, cluster_group;
        """

        # Execute the query and convert the results to a DataFrame
        combined_results = pd.read_sql(cluster_query, engine)

        # Separate DataFrames for each algorithm's counts
        kmeans_counts_html = combined_results[combined_results['algorithm'] == 'KMeans'][['cluster_group', 'count']].dropna().to_html(classes='table table-striped', index=False)
        dbscan_counts_html = combined_results[combined_results['algorithm'] == 'DBSCAN'][['cluster_group', 'count']].dropna().to_html(classes='table table-striped', index=False)
        agglomerative_counts_html = combined_results[combined_results['algorithm'] == 'Agglomerative'][['cluster_group', 'count']].dropna().to_html(classes='table table-striped', index=False)

        # Prepare DataFrames for plotting
        df_kmeans = pd.read_sql("SELECT danceability, energy, kmeans FROM Spotify WHERE kmeans IS NOT NULL", engine)
        df_dbscan = pd.read_sql("SELECT danceability, energy, dbscan FROM Spotify WHERE dbscan IS NOT NULL", engine)
        df_agglomerative = pd.read_sql("SELECT danceability, energy, agglomerative FROM Spotify WHERE agglomerative IS NOT NULL", engine)

        # Generate graphs
        generate_kmeans_graph(df_kmeans)
        generate_dbscan_graph(df_dbscan)
        generate_agglomerative_graph(df_agglomerative)

        # Set paths for graph images
        kmeans_graph = '/static/graphs/kmeans_results.png'
        dbscan_graph = '/static/graphs/dbscan_results.png'
        agglomerative_graph = '/static/graphs/agglomerative_results.png'

        return render_template('comparison.html',
                               kmeans_counts=kmeans_counts_html,
                               dbscan_counts=dbscan_counts_html,
                               agglomerative_counts=agglomerative_counts_html,
                               kmeans_graph=kmeans_graph,
                               dbscan_graph=dbscan_graph,
                               agglomerative_graph=agglomerative_graph)

app.register_blueprint(bp)

def initialize_clustering(uri, engine):
    perform_kmeans_clustering(engine)
    perform_dbscan_clustering(engine)
    perform_agglomerative_clustering(engine)