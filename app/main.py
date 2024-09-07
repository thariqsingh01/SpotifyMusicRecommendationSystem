#main.py

from flask import Blueprint, render_template, request, redirect, url_for
from . import db
from clustering.kmeans import perform_kmeans_clustering
from clustering.dbscan import perform_dbscan_clustering
from clustering.agglomerative import perform_agglomerative_clustering
import pandas as pd
from app.main import SpotifyData

bp = Blueprint('main', __name__)

class SpotifyData(db.Model):
    __tablename__ = 'spotify_data'
    number = db.Column(db.Integer, primary_key=True)
    artist_name = db.Column(db.String)
    track_name = db.Column(db.String)
    track_id = db.Column(db.String)
    popularity = db.Column(db.Integer)
    year = db.Column(db.Integer)
    genre = db.Column(db.String)
    danceability = db.Column(db.Float)
    energy = db.Column(db.Float)
    key = db.Column(db.Integer)
    loudness = db.Column(db.Float)
    mode = db.Column(db.Float)
    speechiness = db.Column(db.Float)
    acousticness = db.Column(db.Float)
    instrumentalness = db.Column(db.Float)
    liveness = db.Column(db.Float)
    valence = db.Column(db.Float)
    tempo = db.Column(db.Float)
    duration_ms = db.Column(db.Integer)
    time_signature = db.Column(db.Integer)
    kmeans = db.Column(db.Integer) 
    dbscan = db.Column(db.Integer)  
    agglomerative = db.Column(db.Integer)  

    def duration_in_minutes_seconds(self):
        if self.duration_ms:
            total_seconds = self.duration_ms // 1000
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes} min {seconds} sec"
        else:
            return None

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

def initialize_clustering():
    perform_kmeans_clustering()
    perform_dbscan_clustering()
    perform_agglomerative_clustering()

@bp.before_first_request
def before_first_request():
    initialize_clustering()