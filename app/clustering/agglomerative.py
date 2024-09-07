import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from app import db
from app.main import SpotifyData

def perform_agglomerative_clustering():
    # Query all data
    data = SpotifyData.query.all()
    
    # Extract features using selected variables
    features = [[d.danceability, d.energy, d.tempo, d.valence] for d in data]
    
    # Perform Agglomerative clustering
    agglomerative = AgglomerativeClustering(n_clusters=10)  # Choose number of clusters
    labels = agglomerative.fit_predict(features)

    # Save cluster labels to the database
    for i, song in enumerate(data):
        song.agglomerative_label = labels[i]
        db.session.add(song)
    db.session.commit()
