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
from dask.distributed import Client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def perform_agglomerative_clustering():
    # Start a Dask client for parallel processing
    client = Client()  # This will create a Dask scheduler and workers
    logger.info("Dask client started for parallel processing.")

    try:
        # Query all data from the Spotify table
        data = SpotifyData.query.all()

        if not data:
            logger.warning("No data retrieved from Spotify table.")
            return

        # Extract features for clustering (danceability, energy, tempo, valence)
        features = [[d.danceability, d.energy, d.tempo, d.valence] for d in data]

        # Convert features to Dask DataFrame for handling large datasets
        dask_df = dd.from_array(features, columns=["danceability", "energy", "tempo", "valence"])

        # Compute the Dask DataFrame to get a NumPy array compatible with sklearn
        features_np = dask_df.compute()

        logger.info(f"Data computed for {len(features_np)} records. Starting Agglomerative Clustering...")

        # Train the scikit-learn Agglomerative Clustering model
        agglomerative = AgglomerativeClustering(n_clusters=10)
        labels = agglomerative.fit_predict(features_np)

        logger.info("Agglomerative Clustering complete. Preparing to update database.")

        # Save cluster labels to the database with batch processing
        song_updates = []
        for i, song in enumerate(data):
            song.agglomerative = labels[i]  # Update each song with its cluster label
            song_updates.append(song)

        # Bulk update to minimize database write operations
        db.session.bulk_save_objects(song_updates)
        db.session.commit()
        logger.info(f"Successfully updated {len(song_updates)} records with Agglomerative Clustering labels.")

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error during Agglomerative Clustering or database update: {e}")
        logger.debug(e, exc_info=True)

    finally:
        # Close the Dask client
        client.close()
        logger.info("Dask client closed after clustering.")

# Call the function if this script is run directly
if __name__ == '__main__':
    perform_agglomerative_clustering()
