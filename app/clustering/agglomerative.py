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
from sklearn.cluster import AgglomerativeClustering
import logging
from app import db
from app.models import SpotifyData
import pandas as pd


def perform_agglomerative_clustering():
    # Query all data from the Spotify table
    data = SpotifyData.query.all()

    if not data:
        logging.warning("No data retrieved from Spotify table.")
        return

    # Extract features for clustering (danceability, energy, tempo, valence)
    features = [[d.danceability, d.energy, d.tempo, d.valence] for d in data]

    # Convert features to Dask DataFrame for handling large datasets
    dask_df = dd.from_array(features, columns=["danceability", "energy", "tempo", "valence"])

    # Convert the Dask DataFrame to NumPy array for sklearn compatibility
    features_np = dask_df.compute()  # Compute the Dask DataFrame to get a NumPy array

    # Train the scikit-learn Agglomerative Clustering model
    agglomerative = AgglomerativeClustering(n_clusters=10)
    labels = agglomerative.fit_predict(features_np)

    # Save cluster labels to the database with batch processing
    try:
        for i, song in enumerate(data):
            song.agglomerative = labels[i]  # Update each song with its cluster label
            db.session.add(song)  # Add the modified song record to the session
        
        # Commit all changes to the database
        db.session.commit()
        logging.info(f"Successfully updated {len(data)} records with Agglomerative Clustering labels.")
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error committing changes to the database: {e}")


