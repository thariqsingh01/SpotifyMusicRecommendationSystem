#agglomerative.py

"""
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from app import db

def perform_agglomerative_clustering():
    from app.models import SpotifyData
    # Query all data
    data = SpotifyData.query.all()
    
    # Extract features using selected variables
    features = [[d.danceability, d.energy, d.tempo, d.valence] for d in data]
    
    # Perform Agglomerative clustering
    agglomerative = AgglomerativeClustering(n_clusters=10)  # Choose number of clusters
    labels = agglomerative.fit_predict(features)

    # Save cluster labels to the database
    for i, song in enumerate(data):
        song.agglomerative = labels[i]
        db.session.add(song)
    db.session.commit()
"""

import dask.dataframe as dd
from dask_ml.cluster import AgglomerativeClustering
from app import db
from app.models import SpotifyData

def perform_agglomerative_clustering():
    # Query all data from the Spotify table
    data = SpotifyData.query.all()
    
    if not data:
        print("No data retrieved from Spotify table.")
        return

    # Extract features for clustering
    features = [[d.danceability, d.energy, d.tempo, d.valence] for d in data]
    
    # Convert features to Dask DataFrame
    dask_df = dd.from_array(features, columns=["danceability", "energy", "tempo", "valence"])

    # Train the Dask-ML Agglomerative Clustering model
    agglomerative = AgglomerativeClustering(n_clusters=10)  # Choose number of clusters
    labels = agglomerative.fit_predict(dask_df)

    # Save cluster labels to the database
    for i, song in enumerate(data):
        song.agglomerative = labels[i]
        db.session.add(song)

    # Commit all changes to the database
    try:
        db.session.commit()
        print(f"Successfully updated {len(data)} records with Dask-ML Agglomerative labels.")
    except Exception as e:
        db.session.rollback()
        print(f"Error committing changes to the database: {e}")
