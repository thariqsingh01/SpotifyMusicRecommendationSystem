#main.py

from flask import Blueprint, render_template, request
from .models import SpotifyData
from app import db
from .clustering.kmeans import perform_kmeans_clustering,retrieve_kmeans_results
from .clustering.dbscan import perform_dbscan_clustering,retrieve_dbscan_results
from .clustering.agglomerative import perform_agglomerative_clustering,retrieve_agglomerative_results
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
    # Get the database engine
    engine = db.engine  # Adjust this line if your db object is named differently

    # Retrieve clustering results
    kmeans_results = retrieve_kmeans_results(engine)
    dbscan_results = retrieve_dbscan_results(engine)
    agglomerative_results = retrieve_agglomerative_results(engine)

    # Convert results to DataFrames
    kmeans_df = kmeans_results[['track_id', 'danceability', 'energy', 'tempo', 'valence', 'kmeans']]
    dbscan_df = dbscan_results[['track_id', 'danceability', 'energy', 'tempo', 'valence', 'dbscan']]
    agglomerative_df = agglomerative_results[['track_id', 'danceability', 'energy', 'tempo', 'valence', 'agglomerative']]

    # Get top 5 recommendations from each clustering technique
    top_kmeans = kmeans_df.groupby('kmeans').head(5)
    top_dbscan = dbscan_df.groupby('dbscan').head(5)
    top_agglomerative = agglomerative_df.groupby('agglomerative').head(5)

    # Convert to HTML for rendering
    top_kmeans_html = top_kmeans.to_html(classes='table table-striped', index=False)
    top_dbscan_html = top_dbscan.to_html(classes='table table-striped', index=False)
    top_agglomerative_html = top_agglomerative.to_html(classes='table table-striped', index=False)

    return render_template('comparison.html',
                           top_kmeans_html=top_kmeans_html,
                           top_dbscan_html=top_dbscan_html,
                           top_agglomerative_html=top_agglomerative_html)


def initialize_clustering(uri, engine):
    perform_kmeans_clustering(uri, engine)
    perform_dbscan_clustering(uri, engine)
    perform_agglomerative_clustering(uri, engine)

