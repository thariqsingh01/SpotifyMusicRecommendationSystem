#kmeans.py

"""
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
    print(f"Type of data: {type(data)}")  

    # Debugging: Check if data is returned correctly
    print("Got to Step 1")
    if not data:
        print("No data retrieved from Spotify table.")
        return

    print("Got to Step 2")
    # Extract features using selected variables
    try:
        features = [[float(d.danceability), float(d.energy), float(d.tempo), float(d.valence)] for d in data]
        #print("features:")
        #print(features)
        #print(f"Type of features: {type(features)}")  # Should be a list of lists
        #if len(features) > 0:
        #    print(f"Type of first feature list: {type(features[0])}")  # Should be a list of floats
        #    print(f"Type of first feature item: {type(features[0][0])}")  # Should be float
    except AttributeError as e:
        print(f"Error accessing attributes: {e}")
        return

    print("Got to Step 3")
    # Perform KMeans clustering
    try:
        kmeans = KMeans(n_clusters=10, random_state=42)
        labels = kmeans.fit_predict(features)
        labels = [float(label) for label in labels]
        #print(f"Type of labels: {type(labels)}")  # Should be a list of floats
        #if len(labels) > 0:
        #    print(f"Type of first label: {type(labels[0])}")  # Should be float
        #print("labels:")
        #print(labels)
    except Exception as e:
        print(f"Error during KMeans clustering: {e}")
        return

    print("Got to Step 4")
    # Save cluster labels to the database
    for i, label in enumerate(labels):
        song_record = SpotifyData.query.get(i)  
        song_record.kmeans = int(label)  
        db.session.add(song_record)

    print("Got to Step 5")
    # Commit all the changes to the database
    try:
        db.session.commit()
        print(f"Successfully updated {len(data)} records with KMeans labels.")
    except Exception as e:
        db.session.rollback()
        print(f"Error committing changes to the database: {e}")
"""

import dask.dataframe as dd
from dask_ml.cluster import KMeans
import pandas as pd
import logging
from app import db
from app.models import SpotifyData

# Set up logging
logging.basicConfig(level=logging.INFO)

def perform_kmeans_clustering():
    # Query all data from the Spotify table
    data = SpotifyData.query.all()
    
    if not data:
        logging.warning("No data retrieved from Spotify table.")
        return

    # Convert data to a Dask DataFrame
    df = dd.from_pandas(pd.DataFrame([{
        "danceability": d.danceability,
        "energy": d.energy,
        "tempo": d.tempo,
        "valence": d.valence,
        "track_id": d.track_id
    } for d in data]), npartitions=4)

    # Train the Dask KMeans model
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(df[['danceability', 'energy', 'tempo', 'valence']])

    # Get cluster labels
    labels = kmeans.predict(df[['danceability', 'energy', 'tempo', 'valence']]).compute()

    # Save cluster labels to the database
    for i, label in enumerate(labels):
        song_record = SpotifyData.query.filter_by(track_id=df['track_id'].loc[i]).first()  
        if song_record:
            song_record.kmeans = int(label)
            db.session.add(song_record)
        else:
            logging.warning(f"No song found with track_id: {df['track_id'].loc[i]}")

    # Commit all the changes to the database
    try:
        db.session.commit()
        logging.info(f"Successfully updated {len(data)} records with Dask KMeans labels.")
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error committing changes to the database: {e}")
