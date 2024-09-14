#main.py

from flask import Blueprint, render_template, request
from .models import SpotifyData
from app import db
from .clustering.kmeans import perform_kmeans_clustering
from .clustering.dbscan import perform_dbscan_clustering
from .clustering.agglomerative import perform_agglomerative_clustering
import pandas as pd


bp = Blueprint('main', __name__)

@bp.route('/search')
def search():
    query = request.args.get('q')
    if query:
        results = SpotifyData.query.filter(
            (SpotifyData.artist_name.ilike(f"%{query}%")) |
            (SpotifyData.track_name.ilike(f"%{query}%")) |
            (SpotifyData.year.ilike(f"%{query}%")) |
            (SpotifyData.genre.ilike(f"%{query}%"))
        ).order_by(SpotifyData.popularity.desc()).limit(20).all()
    else:
        results = []
    return render_template('search.html', results=results)

@bp.route('/')
def home():
    return render_template("app.html")

@bp.route('/comparison')
def comparison():
    # Query all data
    data = SpotifyData.query.all()

    # Extract data for comparison
    kmeans_results = [(song.track_name, song.artist_name, song.kmeans_label) for song in data]
    dbscan_results = [(song.track_name, song.artist_name, song.dbscan_label) for song in data]
    agglomerative_results = [(song.track_name, song.artist_name, song.agglomerative_label) for song in data]

    # Convert to DataFrame for comparison
    kmeans_df = pd.DataFrame(kmeans_results, columns=['Track Name', 'Artist', 'Label'])
    dbscan_df = pd.DataFrame(dbscan_results, columns=['Track Name', 'Artist', 'Label'])
    agglomerative_df = pd.DataFrame(agglomerative_results, columns=['Track Name', 'Artist', 'Label'])

    # Get top 5 recommendations from each clustering technique
    top_kmeans = kmeans_df.groupby('Label').head(5)
    top_dbscan = dbscan_df.groupby('Label').head(5)
    top_agglomerative = agglomerative_df.groupby('Label').head(5)

    # Convert to HTML for rendering
    top_kmeans_html = top_kmeans.to_html(classes='table table-striped')
    top_dbscan_html = top_dbscan.to_html(classes='table table-striped')
    top_agglomerative_html = top_agglomerative.to_html(classes='table table-striped')

    return render_template('comparison.html',
                           top_kmeans_html=top_kmeans_html,
                           top_dbscan_html=top_dbscan_html,
                           top_agglomerative_html=top_agglomerative_html)

def initialize_clustering():
    perform_kmeans_clustering()
    perform_dbscan_clustering()
    perform_agglomerative_clustering()
