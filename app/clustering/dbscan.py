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
from app import db
from app.models import SpotifyData

def perform_dbscan_clustering():
    # Query all data from the Spotify table
    data = SpotifyData.query.all()
    
    if not data:
        print("No data retrieved from Spotify table.")
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
    dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters as needed
    labels = dbscan.fit_predict(features)

    # Save cluster labels to the database
    for i, song in enumerate(data):
        song.dbscan = labels[i]
        db.session.add(song)
    
    # Commit all changes to the database
    try:
        db.session.commit()
        print(f"Successfully updated {len(data)} records with DBSCAN labels.")
    except Exception as e:
        db.session.rollback()
        print(f"Error committing changes to the database: {e}")
