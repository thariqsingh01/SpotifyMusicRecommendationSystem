#dbscan.py

"""
import pandas as pd
from sklearn.cluster import DBSCAN
from app import db

def perform_dbscan_clustering():
    from app.models import SpotifyData
    # Query all data
    data = SpotifyData.query.all()
    
    # Extract features using selected variables
    features = [[d.danceability, d.energy, d.tempo, d.valence] for d in data]
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters as needed
    labels = dbscan.fit_predict(features)

    # Save cluster labels to the database
    for i, song in enumerate(data):
        song.dbscan = labels[i]
        db.session.add(song)
    db.session.commit()

"""

from sklearn.cluster import DBSCAN
import dask.dataframe as dd
import pandas as pd
import logging
from app import db
from app.models import SpotifyData
from dask.distributed import Client

def perform_dbscan_clustering():
    # Start a Dask client for parallel processing
    client = Client()  # This will create a Dask scheduler and workers

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

    # Convert to a numpy array for DBSCAN
    features = df[['danceability', 'energy', 'tempo', 'valence']].compute().values

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust eps and min_samples based on dataset
    labels = dbscan.fit_predict(features)

    # Prepare bulk update for saving cluster labels to the database
    song_updates = []
    for i, song in enumerate(data):
        song.dbscan = labels[i]
        song_updates.append(song)

    # Bulk update the database with the modified records
    try:
        db.session.bulk_save_objects(song_updates)
        db.session.commit()
        logging.info(f"Successfully updated {len(data)} records with DBSCAN labels.")
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error committing changes to the database: {e}")

    finally:
        # Ensure the Dask client is closed properly
        client.close()
