#main.py

from flask import Flask, Blueprint, render_template, request, current_app, jsonify
from .models import SpotifyData
from app import db
from .clustering.kmeans import perform_kmeans_clustering, generate_kmeans_graph
from .clustering.dbscan import perform_dbscan_clustering, generate_dbscan_graph
from .clustering.agglomerative import perform_agglomerative_clustering, generate_agglomerative_graph
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy import create_engine ,inspect,text
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
from sklearn.decomposition import PCA
import dask.dataframe as dd
from sklearn.preprocessing import StandardScaler
from dask import delayed, compute
from dask.distributed import Client
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


from flask import Blueprint, render_template, current_app
from flask_caching import Cache
from sqlalchemy import inspect, text
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, fowlkes_mallows_score
from sklearn.decomposition import PCA
import logging

# Initialize the logger
logger = logging.getLogger(__name__)

# Blueprint and cache setup
bp = Blueprint('comparison', __name__)
cache = Cache()

def initialize_clustering(uri, engine):
    perform_kmeans_clustering(engine)
    perform_dbscan_clustering(engine)
    perform_agglomerative_clustering(engine)

@cache.cached(timeout=300)
@bp.route('/comparison', methods=['GET', 'POST'])
def comparison():
    with current_app.app_context():
        # Get the connection string from the SQLAlchemy engine
        engine = db.engine
        connection_string = engine.url.__to_string__(hide_password=False)

        # Inspect the database to check if the table exists
        inspector = inspect(engine)
        if 'Spotify' not in inspector.get_table_names():
            logger.error("Spotify table does not exist.")
            return "Error: The 'Spotify' table does not exist in the database. Please ensure the database is set up correctly.", 404 

        query = """
            SELECT track_id, danceability, energy, kmeans, dbscan, agglomerative
            FROM Spotify
            WHERE kmeans IS NOT NULL OR dbscan IS NOT NULL OR agglomerative IS NOT NULL;
        """

        with engine.connect() as connection:
            result = connection.execute(text(query))
            data = result.fetchall()
            if not data:
                logger.error("No data available for clustering.")
                return "Error: No data available for clustering. Please check your database entries.", 404 

        df_features = pd.DataFrame(data, columns=result.keys())

        for col in ['danceability', 'energy', 'kmeans', 'dbscan', 'agglomerative']:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

        df_features = df_features.dropna()

        if df_features.empty:
            logger.warning("No valid data left for clustering after dropping NaNs.")
            return "Error: Not enough valid data for clustering. Ensure that your dataset contains valid entries.", 400

        # Log data shape after cleaning
        logger.info(f"Data shape after dropping NaNs: {df_features.shape}")

        # Sample 10% of the dataset
        df_features = df_features.sample(frac=0.1, random_state=1)
        logger.info(f"Data shape after sampling 10%: {df_features.shape}")

        # Standardize the data before applying PCA
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_features[['danceability', 'energy']])
        logger.info(f"Scaled features shape: {scaled_features.shape}")

        # Apply PCA to reduce dimensionality
        try:
            pca_features = apply_pca(pd.DataFrame(scaled_features), n_components=2)
            logger.info(f"PCA output shape: {pca_features.shape}")
        except Exception as e:
            logger.error(f"Error during PCA: {e}")
            return "Error: PCA failed. Please check your data.", 500

        # Reshape data to count clusters by algorithm
        df_melted = df_features.melt(
            id_vars=['danceability', 'energy'],
            value_vars=['kmeans', 'dbscan', 'agglomerative'],
            var_name='algorithm',
            value_name='cluster_group'
        ).groupby(['algorithm', 'cluster_group']).size().reset_index()

        df_melted.columns = ['algorithm', 'cluster_group', 'count']
        logger.info(f"Cluster count dataframe: {df_melted.head()}")

        # Generate graphs and calculate metrics using PCA features
        try:
            kmeans_graph = generate_kmeans_graph(pd.DataFrame(pca_features, columns=['pca1', 'pca2']), df_features['kmeans'])
            dbscan_graph = generate_dbscan_graph(pd.DataFrame(pca_features, columns=['pca1', 'pca2']), df_features['dbscan'])
            agglomerative_graph = generate_agglomerative_graph(pd.DataFrame(pca_features, columns=['pca1', 'pca2']), df_features['agglomerative'])

            logger.info("Graphs generated successfully.")

            metrics = calculate_metrics(df_features)
            logger.info(f"Metrics calculated: {metrics}")
        except Exception as e:
            logger.error(f"Error during graph generation or metrics calculation: {e}")
            return "Error: There was an issue generating the graphs or calculating the metrics. Please check your data.", 500

        # Set paths for graph images
        graph_paths = {
            'kmeans': '/static/graphs/kmeans_results.png',
            'dbscan': '/static/graphs/dbscan_results.png',
            'agglomerative': '/static/graphs/agglomerative_results.png'
        }

        # Render the template with the calculated metrics and graph paths
        return render_template('comparison.html',
                               kmeans_counts=df_melted[df_melted['algorithm'] == 'kmeans'][['cluster_group', 'count']].to_html(classes='table table-striped', index=False),
                               dbscan_counts=df_melted[df_melted['algorithm'] == 'dbscan'][['cluster_group', 'count']].to_html(classes='table table-striped', index=False),
                               agglomerative_counts=df_melted[df_melted['algorithm'] == 'agglomerative'][['cluster_group', 'count']].to_html(classes='table table-striped', index=False),
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
    metrics = {}
    logger.info("Starting metrics calculation.")

    for algo in ['kmeans', 'dbscan', 'agglomerative']:
        if algo in df.columns:
            features = df[['danceability', 'energy']]
            labels = df[algo]
            logger.info(f"Calculating metrics for {algo} with labels: {labels.unique()}")

            if has_enough_data(labels):
                silhouette = calculate_silhouette_score(features, labels)
                davies_bouldin = calculate_davies_bouldin_score(features, labels)
                calinski_harabasz = calculate_calinski_harabasz_index(features, labels)
                wcss = calculate_wcss(features, labels)
                fmi = calculate_fmi(df['track_id'], labels)

                metrics[f'{algo.capitalize()} Silhouette Score'] = silhouette
                metrics[f'{algo.capitalize()} Davies-Bouldin Index'] = davies_bouldin
                metrics[f'{algo.capitalize()} Calinski-Harabasz Index'] = calinski_harabasz
                metrics[f'{algo.capitalize()} WCSS'] = wcss
                metrics[f'{algo.capitalize()} Fowlkes-Mallows Index'] = fmi

                logger.info(f"{algo.capitalize()} metrics: Silhouette={silhouette}, Davies-Bouldin={davies_bouldin}, Calinski-Harabasz={calinski_harabasz}, WCSS={wcss}, FMI={fmi}")
            else:
                metrics[f'{algo.capitalize()} Metrics'] = 'Not enough data'
                logger.warning(f"Not enough data for {algo.capitalize()} metrics.")

    return metrics

# Supporting functions for metrics
def has_enough_data(labels):
    """Checks if the cluster labels have enough data for metric calculation."""
    return not labels.isnull().all() and len(labels.unique()) > 1

def calculate_silhouette_score(data, labels):
    """Calculates the Silhouette Score."""
    return silhouette_score(data, labels) if has_enough_data(labels) else None

def calculate_davies_bouldin_score(data, labels):
    """Calculates the Davies-Bouldin Index."""
    return davies_bouldin_score(data, labels) if has_enough_data(labels) else None

def calculate_calinski_harabasz_index(data, labels):
    """Calculates the Calinski-Harabasz Index."""
    return calinski_harabasz_score(data, labels) if has_enough_data(labels) else None

def calculate_wcss(data, labels):
    """Calculates Within-Cluster Sum of Squares (WCSS) for the clustering."""
    centroids = np.array([data[labels == i].mean(axis=0) for i in np.unique(labels)])
    distances = np.sum([np.sum((data[labels == i] - centroids[i]) ** 2) for i in np.unique(labels)])
    return distances if has_enough_data(labels) else None

def calculate_fmi(true_labels, predicted_labels):
    """Calculates the Fowlkes-Mallows Index."""
    return fowlkes_mallows_score(true_labels, predicted_labels) if has_enough_data(predicted_labels) else None

def apply_pca(data, n_components=None):
    if n_components is None or n_components > min(data.shape):
        n_components = min(data.shape)

    logger.info(f"Applying PCA with {n_components} components on data with shape {data.shape}.")
    pca = PCA(n_components=n_components)
    try:
        result = pca.fit_transform(data)
        logger.info(f"PCA completed successfully. Result shape: {result.shape}")
        return result
    except Exception as e:
        logger.error(f"PCA failed with error: {e}")
        raise
