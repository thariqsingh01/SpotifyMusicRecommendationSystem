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
from flask_debugtoolbar import DebugToolbarExtension

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# After initializing your Flask app
app.config['DEBUG_TB_INTERCEPT_REDIRECTS'] = False
toolbar = DebugToolbarExtension(app)

# Set up Spotipy with your Spotify API credentials
client_id = os.getenv('SPOTIFY_CLIENT_ID', 'default_client_id')
client_secret = os.getenv('SPOTIFY_CLIENT_SECRET', 'default_client_secret')
credentials = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=credentials)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

bp = Blueprint('main', __name__)
#app.register_blueprint(bp)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q')
    logger.info(f'Full request URL: {request.url}')  # Log the full request URL
    logger.info(f'Search query received: {query}')  # Log the search query
    
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
    return render_template('partials/_search_results.html', results=results)

def get_recommendations(user_choices):
    # Fetch recommendations based on user choices from the database
    recommendations = {'KMeans': [], 'DBSCAN': [], 'Agglomerative': []}
    
    for choice in user_choices:
        selected_song = SpotifyData.query.filter_by(track_id=choice).first()
        if selected_song:
            for algorithm in ['kmeans', 'dbscan', 'agglomerative']:
                cluster = getattr(selected_song, algorithm)
                if cluster is not None:
                    similar_songs = SpotifyData.query.filter(getattr(SpotifyData, algorithm) == cluster).limit(5).all()
                    for song in similar_songs:
                        # Fetch album cover from Spotify
                        track_info = sp.track(song.track_id)  # Assuming track_id is Spotify's track ID
                        album_cover = track_info['album']['images'][0]['url'] if track_info['album']['images'] else None
                        recommendations[algorithm.capitalize()].append({
                            'track_id': song.track_id,
                            'artist_name': song.artist_name,
                            'track_name': song.track_name,
                            'year': song.year,
                            'genre': song.genre,
                            'duration': song.duration_in_minutes_seconds(),
                            'album_cover': album_cover  # Add album cover URL
                        })
        else:
            logger.warning(f'Selected song not found for track_id: {choice}')
    return recommendations

@bp.route('/suggestions', methods=['GET'])
def suggestions():
    track_id = request.args.get('track_id')
    logger.info(f'Suggestions request for track_id: {track_id}')  # Log the request
    if not track_id:
        logger.error('Track ID is required for suggestions.')
        return {'error': 'Track ID is required.'}, 400

    # Find the KMeans cluster for the selected track
    selected_track = SpotifyData.query.filter_by(track_id=track_id).first()
    if not selected_track or selected_track.kmeans is None:
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

def initialize_clustering(uri, engine):
    perform_kmeans_clustering()
    perform_dbscan_clustering()
    perform_agglomerative_clustering()
