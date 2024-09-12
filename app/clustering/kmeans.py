#kmeans.py

import pandas as pd
from sklearn.cluster import KMeans
from app import db
from app.models import SpotifyData

def perform_kmeans_clustering():
    # Query all data
    data = SpotifyData.query.all()
    
    # Extract features using selected variables
    features = [[d.danceability, d.energy, d.tempo, d.valence] for d in data]
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=10)  # Choose number of clusters
    labels = kmeans.fit_predict(features)

    # Save cluster labels to the database
    for i, song in enumerate(data):
        song.kmeans = labels[i]
        db.session.add(song)
    db.session.commit()
