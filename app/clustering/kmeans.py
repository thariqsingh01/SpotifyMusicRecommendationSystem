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
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import pandas as pd
import logging
from app import db
from app.models import SpotifyData
import dask.array as da
from sklearn.preprocessing import StandardScaler  # For scaling data before clustering


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

    # Extract relevant features for clustering
    features = df[['danceability', 'energy', 'tempo', 'valence']].compute()

    # Standardize the features for better clustering performance
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 1. KMeans Clustering
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(features)
    kmeans_labels = kmeans.predict(features).compute()

    # 2. DBSCAN Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    dbscan_labels = dbscan.fit_predict(scaled_features)  # Use scaled features for DBSCAN

    # 3. Agglomerative Clustering
    agglomerative = AgglomerativeClustering(n_clusters=10)
    agglomerative_labels = agglomerative.fit_predict(scaled_features)

    # Convert all label arrays to Dask Arrays
    kmeans_labels_array = da.from_array(kmeans_labels, chunks=(len(kmeans_labels) // 4,))
    dbscan_labels_array = da.from_array(dbscan_labels, chunks=(len(dbscan_labels) // 4,))
    agglomerative_labels_array = da.from_array(agglomerative_labels, chunks=(len(agglomerative_labels) // 4,))

    # Assign labels to the DataFrame
    df['kmeans_cluster'] = kmeans_labels_array
    df['dbscan_cluster'] = dbscan_labels_array
    df['agglomerative_cluster'] = agglomerative_labels_array

    # Save cluster labels to the database using batch processing
    for track_id, kmeans_label, dbscan_label, agglomerative_label in zip(
        df['track_id'].compute(), 
        kmeans_labels, 
        dbscan_labels, 
        agglomerative_labels):
        
        song_record = SpotifyData.query.filter_by(track_id=track_id).first()
        if song_record:
            song_record.kmeans = int(kmeans_label)
            song_record.dbscan = int(dbscan_label)
            song_record.agglomerative = int(agglomerative_label)
            db.session.add(song_record)
        else:
            logging.warning(f"No song found with track_id: {track_id}")

    # Commit all the changes to the database
    try:
        db.session.commit()
        logging.info(f"Successfully updated {len(data)} records with clustering labels.")
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error committing changes to the database: {e}")
