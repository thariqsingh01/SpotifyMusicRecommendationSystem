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

from sklearn.cluster import KMeans
import dask.dataframe as dd
import pandas as pd
import logging
from app import db
from app.models import SpotifyData
from dask.distributed import Client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def perform_kmeans_clustering():
    # Start a Dask client for parallel processing
    client = Client()  # This will create a Dask scheduler and workers
    logger.info("Dask client started for parallel processing.")

    try:
        # Query all data from the Spotify table
        data = SpotifyData.query.all()

        if not data:
            logger.warning("No data retrieved from Spotify table.")
            return

        # Convert queried data to a pandas DataFrame
        df = pd.DataFrame([{
            "danceability": d.danceability,
            "energy": d.energy,
            "tempo": d.tempo,
            "valence": d.valence,
            "track_id": d.track_id
        } for d in data])

        logger.info(f"Data retrieved: {len(df)} rows from Spotify table.")

        # Ensure that the DataFrame is UTF-8 encoded to avoid UnicodeDecodeError
        df = df.applymap(lambda x: x.encode('utf-8', errors='ignore').decode('utf-8') if isinstance(x, str) else x)

        # Convert pandas DataFrame to Dask DataFrame for distributed processing
        npartitions = 10  # Increase the partition count for better parallelization
        ddf = dd.from_pandas(df, npartitions=npartitions)

        # Scatter the DataFrame across Dask workers to optimize processing
        scattered_data = client.scatter(ddf)

        # Extract features for clustering
        features = scattered_data[['danceability', 'energy', 'tempo', 'valence']].compute().values

        # Perform KMeans clustering
        logger.info("Performing KMeans clustering...")
        kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust n_clusters as needed
        labels = kmeans.fit_predict(features)

        # Prepare the bulk update of cluster labels back to the database
        song_updates = []
        for i, song in enumerate(data):
            song.kmeans = labels[i]  # Assuming 'kmeans' is the field to store the cluster label
            song_updates.append(song)

        # Bulk update the database with the new cluster labels
        db.session.bulk_save_objects(song_updates)
        db.session.commit()
        logger.info(f"Successfully updated {len(song_updates)} records with KMeans labels.")

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error during KMeans clustering or database update: {e}")
        logger.debug(e, exc_info=True)

    finally:
        # Close the Dask client
        client.close()
        logger.info("Dask client closed after clustering.")

# Call the function if this script is run directly
if __name__ == '__main__':
    perform_kmeans_clustering()
