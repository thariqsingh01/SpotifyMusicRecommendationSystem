# main.py

from flask import Blueprint, render_template, request
from .models import SpotifyData
from app import db
from .clustering.kmeans import perform_kmeans_clustering, generate_kmeans_graph
from .clustering.dbscan import perform_dbscan_clustering, generate_dbscan_graph
from .clustering.agglomerative import perform_agglomerative_clustering, generate_agglomerative_graph
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

@bp.route('/comparison', methods=['GET', 'POST'])
def comparison():
    if request.method == 'POST':
        # Get the database engine
        engine = db.engine

        #generate graphs
        generate_kmeans_graph()
        generate_dbscan_graph()
        generate_agglomerative_graph()

        # Retrieve clustering results
        kmeans_results = pd.read_sql("SELECT track_id, danceability, energy, tempo, valence, kmeans FROM Spotify WHERE kmeans IS NOT NULL", engine)
        dbscan_results = pd.read_sql("SELECT track_id, danceability, energy, tempo, valence, dbscan FROM Spotify WHERE dbscan IS NOT NULL", engine)
        agglomerative_results = pd.read_sql("SELECT track_id, danceability, energy, tempo, valence, agglomerative FROM Spotify WHERE agglomerative IS NOT NULL", engine)

        # Convert results to DataFrames
        top_kmeans = kmeans_results.groupby('kmeans').head(5)
        top_dbscan = dbscan_results.groupby('dbscan').head(5)
        top_agglomerative = agglomerative_results.groupby('agglomerative').head(5)

        # Convert to HTML for rendering
        top_kmeans_html = top_kmeans.to_html(classes='table table-striped', index=False)
        top_dbscan_html = top_dbscan.to_html(classes='table table-striped', index=False)
        top_agglomerative_html = top_agglomerative.to_html(classes='table table-striped', index=False)

        return render_template('comparison.html',
                               top_kmeans_html=top_kmeans_html,
                               top_dbscan_html=top_dbscan_html,
                               top_agglomerative_html=top_agglomerative_html,
                               kmeans_graph='static/graphs/kmeans_results.png',
                               dbscan_graph='static/graphs/dbscan_results.png',
                               agglomerative_graph='static/graphs/agglomerative_results.png')

    return render_template('comparison.html')  # For GET requests, just render an empty comparison page

def initialize_clustering(uri, engine):
    perform_kmeans_clustering()
    perform_dbscan_clustering()
    perform_agglomerative_clustering()
