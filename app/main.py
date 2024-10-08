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
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Set up Spotipy with your Spotify API credentials
client_id = os.getenv('SPOTIFY_CLIENT_ID', 'default_client_id')
client_secret = os.getenv('SPOTIFY_CLIENT_SECRET', 'default_client_secret')
credentials = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=credentials)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

bp = Blueprint('main', __name__)

def get_user_choices(request_data):
    """
    Extracts user-selected songs from the submitted JSON data.
    Args:    request_data (dict): A list of dictionaries containing song data submitted by the user.
    Returns: list: A list of SpotifyData objects representing user-selected songs.
    """
    user_choices = []
    for item in request_data:
        if item.get('track_name') and item.get('artist_name'):  # Ensure both song and artist are present
            song = SpotifyData.query.filter_by(track_name=item['track_name'], artist_name=item['artist_name']).first()
            if song:
                user_choices.append(song)
                logger.info(f'This is the current song: {song}')
    return user_choices

def calculate_similarity(user_features, song_features):
    """
    Calculates the cosine similarity between the user's selected song features and a given song's features.
    Args:  user_features (list): A list of features from the user's selected song (e.g., danceability, energy).
           song_features (list): A list of features from a Spotify song.
    Returns:float: The cosine similarity score between the user's song and the given song.
    """
    similarity = cosine_similarity([user_features], [song_features])[0][0]
    return similarity

def generate_recommendations(user_song, cluster_label_field):
    """
    Generates recommendations for the user based on their selected song and assigned cluster label.
    Args:
        user_song (SpotifyData object): The song chosen by the user.
        cluster_label_field (str): The name of the cluster label field (e.g., 'kmeans', 'dbscan', 'agglomerative').
    
    Returns:
        list: A list of recommended songs based on similarity and cluster membership.
    """
    # Query for songs in the same cluster as the user's song
    similar_songs = SpotifyData.query.filter(getattr(SpotifyData, cluster_label_field) == getattr(user_song, cluster_label_field)).all()

    # Extract relevant features from the user's selected song
    user_features = [user_song.danceability, user_song.energy, user_song.tempo, user_song.valence]

    recommendations = []
    for song in similar_songs:
        # Extract the features of each song in the same cluster
        song_features = [song.danceability, song.energy, song.tempo, song.valence]
        
        # Calculate similarity score between the user's song and the current song
        similarity_score = calculate_similarity(user_features, song_features)
        
        # Append the song and its similarity score to the recommendations list
        recommendations.append((song, similarity_score))

    # Sort recommendations by similarity score in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Get the top 5 recommendations
    top_recommendations = [
        {
            'track_id': song.track_id,
            'artist_name': song.artist_name,
            'track_name': song.track_name,
            'year': song.year,
            'genre': song.genre,
            'cover_url': song.cover_url 
        }
        for song, _ in recommendations[:5]
    ]

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
        user_cluster_label = user_song.kmeans  # KMeans label
        dbscan_cluster_label = user_song.dbscan  # DBSCAN label
        agglomerative_cluster_label = user_song.agglomerative  # Agglomerative label

        # Validate the cluster labels
        if user_cluster_label is None or dbscan_cluster_label is None or agglomerative_cluster_label is None:
            return jsonify({"message": "Cluster labels not found"}), 404

        # Generate recommendations for each clustering algorithm
        kmeans_recommendations = generate_recommendations(user_song, 'kmeans')
        dbscan_recommendations = generate_recommendations(user_song, 'dbscan')
        agglomerative_recommendations = generate_recommendations(user_song, 'agglomerative')

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