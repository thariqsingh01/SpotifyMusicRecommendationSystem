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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def perform_dbscan_clustering():
    # Start a Dask client for parallel processing
    client = Client()  # This will create a Dask scheduler and workers
    logger.info("Dask client started for parallel processing.")

    try:
        # Query all data from the Spotify table
        data = SpotifyData.query.all()

        if not data:
            logger.warning("No data retrieved from Spotify table.")
            return

        # Convert queried data to a Dask DataFrame
        df = dd.from_pandas(pd.DataFrame([{
            "danceability": d.danceability,
            "energy": d.energy,
            "tempo": d.tempo,
            "valence": d.valence,
            "track_id": d.track_id
        } for d in data]), npartitions=4)

        # Compute the Dask DataFrame to a NumPy array for DBSCAN
        features = df[['danceability', 'energy', 'tempo', 'valence']].compute().values
        logger.info(f"Data computed for {len(features)} records. Starting DBSCAN clustering...")

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust eps and min_samples based on dataset
        labels = dbscan.fit_predict(features)

        logger.info("DBSCAN clustering complete. Preparing to update database.")

        # Prepare bulk update for saving cluster labels to the database
        song_updates = []
        for i, song in enumerate(data):
            song.dbscan = labels[i]  # Update each song with its cluster label
            song_updates.append(song)

        # Bulk update the database with the modified records
        db.session.bulk_save_objects(song_updates)
        db.session.commit()
        logger.info(f"Successfully updated {len(song_updates)} records with DBSCAN labels.")

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error during DBSCAN clustering or database update: {e}")
        logger.debug(e, exc_info=True)

    finally:
        # Ensure the Dask client is closed properly
        client.close()
        logger.info("Dask client closed after clustering.")

# Call the function if this script is run directly
if __name__ == '__main__':
    perform_dbscan_clustering()
