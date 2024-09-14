import pandas as pd
from sklearn.cluster import KMeans
from app import db
import numpy as np

def perform_kmeans_clustering():
    from app.models import SpotifyData

    # Query all data from the Spotify table
    data = SpotifyData.query.all()
    print("data:")
    print(data)
    print(f"Type of data: {type(data)}")  # Should be a list of SpotifyData objects

    # Debugging: Check if data is returned correctly
    print("Got to Step 1")
    if not data:
        print("No data retrieved from Spotify table.")
        return

    print("Got to Step 2")
    # Extract features using selected variables
    try:
        features = [[float(d.danceability), float(d.energy), float(d.tempo), float(d.valence)] for d in data]
        print("features:")
        print(features)
        print(f"Type of features: {type(features)}")  # Should be a list of lists
        if len(features) > 0:
            print(f"Type of first feature list: {type(features[0])}")  # Should be a list of floats
            print(f"Type of first feature item: {type(features[0][0])}")  # Should be float
    except AttributeError as e:
        print(f"Error accessing attributes: {e}")
        return

    print("Got to Step 3")
    # Perform KMeans clustering
    try:
        kmeans = KMeans(n_clusters=10, random_state=42)
        labels = kmeans.fit_predict(features)
        labels = [float(label) for label in labels]
        print(f"Type of labels: {type(labels)}")  # Should be a list of floats
        if len(labels) > 0:
            print(f"Type of first label: {type(labels[0])}")  # Should be float
        print("labels:")
        print(labels)
    except Exception as e:
        print(f"Error during KMeans clustering: {e}")
        return

    print("Got to Step 4")
    # Save cluster labels to the database
    for i, song in enumerate(data):
        print(f"Updating song {song.track_name}: kmeans={int(labels[i])} (type: {type(int(labels[i]))})")
        song.kmeans = int(labels[i])

    print("Got to Step 5")
    # Commit all the changes to the database
    try:
        db.session.commit()
        print(f"Successfully updated {len(data)} records with KMeans labels.")
    except Exception as e:
        db.session.rollback()
        print(f"Error committing changes to the database: {e}")
