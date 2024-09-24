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

import h2o
from h2o.estimators import H2OHierarchicalEstimator
from app import db

def perform_agglomerative_clustering():
    from app.models import SpotifyData
    
    # Initialize H2O with a memory limit
    h2o.init(max_mem_size="4G")  # Adjust based on your system's capabilities

    # Query all data from the Spotify table
    data = SpotifyData.query.all()

    if not data:
        print("No data retrieved from Spotify table.")
        return

    # Extract features for clustering
    features = [[d.danceability, d.energy, d.tempo, d.valence] for d in data]
    
    # Convert features to H2O Frame
    h2o_df = h2o.H2OFrame(features, column_names=["danceability", "energy", "tempo", "valence"])

    # Train the H2O Hierarchical Clustering model
    agglomerative = H2OHierarchicalEstimator(k=10)  # Choose number of clusters
    agglomerative.train(training_frame=h2o_df)

    # Get the cluster labels
    labels = agglomerative.predict(h2o_df)

    # Save cluster labels to the database
    for i, song in enumerate(data):
        song.agglomerative = int(labels.as_data_frame()['predict'][i])
        db.session.add(song)
    
    # Commit all changes to the database
    try:
        db.session.commit()
        print(f"Successfully updated {len(data)} records with H2O Agglomerative labels.")
    except Exception as e:
        db.session.rollback()
        print(f"Error committing changes to the database: {e}")

    # Shutdown H2O instance
    h2o.shutdown()
