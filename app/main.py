#main.py

from flask import Flask, Blueprint, render_template, request, current_app, jsonify
from .models import SpotifyData
from app import db
from .clustering.kmeans import perform_kmeans_clustering, generate_kmeans_graph
from .clustering.dbscan import perform_dbscan_clustering, generate_dbscan_graph
from .clustering.agglomerative import perform_agglomerative_clustering, generate_agglomerative_graph
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy import create_engine 
import pandas as pd
import numpy as np
import logging
from flask_caching import Cache
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import mysql.connector
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances

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
        if track and 'album' in track and 'images' in track['album']:
            return track['album']['images'][0]['url'] if track['album']['images'] else None
        else:
            logger.warning(f"No album or images found for track ID: {track_id}")
            return None
    except spotipy.exceptions.SpotifyException as e:
        logger.error(f"Spotify API error for track_id {track_id}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching cover for track_id {track_id}: {str(e)}")
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

@bp.route('/suggestions', methods=['POST'])
def suggestions():
    data = request.get_json()  # Parse JSON data from the request body
    track_name = data.get('track_name')
    artist_name = data.get('artist_name')

    logging.info(f"Received request for suggestions: {data}")

    try:
        # First, try to find the selected song
        selected_song = SpotifyData.query.filter_by(track_name=track_name, artist_name=artist_name).first()

        if not selected_song:
            logging.warning(f"No matching song found for track: {track_name}, artist: {artist_name}")

            # If no matching song is found, suggest the next best song from the database
            next_best_song = SpotifyData.query.order_by(SpotifyData.popularity.desc()).first()

            if not next_best_song:
                logging.warning("No songs available in the database.")
                return jsonify({'error': 'No songs available to suggest'}), 404
            
            logging.info(f"Suggesting next best song: {next_best_song.track_name} by {next_best_song.artist_name}")

            # Use the next best song's cluster to find similar songs
            similar_songs = SpotifyData.query.filter(
                SpotifyData.kmeans == next_best_song.kmeans  # Adjust according to your logic
            ).limit(10).all()

        else:
            logging.info(f"Found selected song: {selected_song.track_name} by {selected_song.artist_name}")

            # Use the selected song's cluster to find similar songs
            similar_songs = SpotifyData.query.filter(
                SpotifyData.kmeans == selected_song.kmeans  # You can switch to dbscan/agglomerative if needed
            ).limit(10).all()

        if not similar_songs:
            logging.warning("No similar songs found for the selected or fallback song.")
            return jsonify({'error': 'No similar songs found'}), 404

        # Prepare the response data with relevant details (track name, artist, genre, etc.)
        suggestions_list = [{
            'track_name': song.track_name,
            'artist_name': song.artist_name,
            'genre': song.genre,
            'year': song.year,
            'duration': f"{song.duration_ms // 60000}min {((song.duration_ms // 1000) % 60)}sec" 
        } for song in similar_songs]

        logging.info(f"Returning {len(suggestions_list)} suggestions")

        return jsonify({'suggestions': suggestions_list})

    except Exception as e:
        logging.error(f"Error while fetching suggestions: {e}")
        return jsonify({'error': 'Failed to fetch suggestions'}), 500


@bp.route('/')
def home():
    return render_template("app.html")

@cache.cached(timeout=300)
@bp.route('/comparison', methods=['GET', 'POST'])
def comparison():
    with current_app.app_context():
        engine = db.engine
        
        # Query to fetch relevant data only once
        query = """
            SELECT danceability, energy, kmeans, dbscan, agglomerative
            FROM Spotify
            WHERE kmeans IS NOT NULL OR dbscan IS NOT NULL OR agglomerative IS NOT NULL;
        """
        df_features = pd.read_sql(query, engine).dropna()

        # Reshape data to count clusters by algorithm
        cluster_counts = df_features.melt(
            id_vars=['danceability', 'energy'],
            value_vars=['kmeans', 'dbscan', 'agglomerative'],
            var_name='algorithm',
            value_name='cluster_group'
        ).groupby(['algorithm', 'cluster_group']).size().reset_index(name='count')

        # Create ProcessPoolExecutor for parallel tasks
        with ProcessPoolExecutor() as executor:
            # Submit tasks for graph generation and metrics calculation
            futures = {
                'kmeans_graph': executor.submit(generate_kmeans_graph, df_features[['danceability', 'energy', 'kmeans']]),
                'dbscan_graph': executor.submit(generate_dbscan_graph, df_features[['danceability', 'energy', 'dbscan']]),
                'agglomerative_graph': executor.submit(generate_agglomerative_graph, df_features[['danceability', 'energy', 'agglomerative']]),
                'metrics': executor.submit(calculate_metrics, df_features)
            }

            # Retrieve results from futures
            graphs = {name: futures[name].result() for name in ['kmeans_graph', 'dbscan_graph', 'agglomerative_graph']}
            metrics = futures['metrics'].result()

        # Set paths for graph images
        graph_paths = {
            'kmeans': '/static/graphs/kmeans_results.png',
            'dbscan': '/static/graphs/dbscan_results.png',
            'agglomerative': '/static/graphs/agglomerative_results.png'
        }

        # Render the template with the calculated metrics and graph paths
        return render_template('comparison.html',
                               kmeans_counts=cluster_counts[cluster_counts['algorithm'] == 'kmeans'][['cluster_group', 'count']].to_html(classes='table table-striped', index=False),
                               dbscan_counts=cluster_counts[cluster_counts['algorithm'] == 'dbscan'][['cluster_group', 'count']].to_html(classes='table table-striped', index=False),
                               agglomerative_counts=cluster_counts[cluster_counts['algorithm'] == 'agglomerative'][['cluster_group', 'count']].to_html(classes='table table-striped', index=False),
                               kmeans_graph=graph_paths['kmeans'],
                               dbscan_graph=graph_paths['dbscan'],
                               agglomerative_graph=graph_paths['agglomerative'],
                               metrics=metrics)


app.register_blueprint(bp)

def initialize_clustering(uri, engine):
    perform_kmeans_clustering(engine)
    perform_dbscan_clustering(engine)
    perform_agglomerative_clustering(engine)

def calculate_metrics(df):
    """Calculates evaluation metrics for clustering results."""
    df.dropna(subset=['kmeans', 'dbscan', 'agglomerative'], how='all', inplace=True)

    metrics = {}

    # Calculate metrics for each clustering algorithm
    for algo in ['kmeans', 'dbscan', 'agglomerative']:
        if algo in df.columns:
            features = df[['danceability', 'energy']]
            labels = df[algo]
            if has_enough_data(labels):
                silhouette = calculate_silhouette_score(features, labels)
                davies_bouldin = calculate_davies_bouldin_score(features, labels)
                dunn = calculate_dunn_index(features, labels)

                metrics[f'{algo.capitalize()} Silhouette Score'] = silhouette
                metrics[f'{algo.capitalize()} Davies-Bouldin Index'] = davies_bouldin
                metrics[f'{algo.capitalize()} Dunn Index'] = dunn
            else:
                metrics[f'{algo.capitalize()} Metrics'] = 'Not enough data'

    return metrics

# Check if there's enough data for metric calculation
def has_enough_data(labels):
    """Checks if the cluster labels have enough data for metric calculation."""
    return not labels.isnull().all() and len(labels.unique()) > 1

# Calculate Silhouette Score
def calculate_silhouette_score(data, labels):
    """Calculates the Silhouette Score."""
    return silhouette_score(data, labels) if has_enough_data(labels) else None

# Calculate Davies-Bouldin Index
def calculate_davies_bouldin_score(data, labels):
    """Calculates the Davies-Bouldin Index."""
    return davies_bouldin_score(data, labels) if has_enough_data(labels) else None

# Optimized Dunn Index calculation
def calculate_dunn_index(data, labels):
    """Calculates the Dunn Index."""
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2:
        return None

    # Precompute pairwise distances for all data points
    distances = pairwise_distances(data, data)

    # Compute intra-cluster distances
    intra_cluster_distances = [
        np.max(distances[np.ix_(labels == label, labels == label)])
        for label in unique_labels
    ]

    # Compute inter-cluster distances
    inter_cluster_distances = [
        np.min(distances[np.ix_(labels == i, labels == j)])
        for i in unique_labels for j in unique_labels if i != j
    ]

    # Avoid division by zero
    max_intra = np.max(intra_cluster_distances)
    if max_intra == 0:
        return None

    return np.min(inter_cluster_distances) / max_intra

def pairwise_distances(X1, X2):
    """Calculate the pairwise distance between two sets of points."""
    return cdist(X1, X2, metric='euclidean')