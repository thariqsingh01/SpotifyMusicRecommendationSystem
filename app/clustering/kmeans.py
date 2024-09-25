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


import logging
from sklearn.cluster import KMeans
import dask.dataframe as dd
import pandas as pd
from app import db
from app.models import SpotifyData
import numpy as np
from dask.distributed import Client
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_kmeans_clustering(uri, engine):
    client = Client()
    logger.info("Dask client started for parallel processing.")

    try:
        # Retrieve data from Spotify table
        data = SpotifyData.query.all()

        if not data:
            logger.warning("No data retrieved from Spotify table.")
            return

        # Convert data to pandas DataFrame
        df = pd.DataFrame([{
            "danceability": d.danceability,
            "energy": d.energy,
            "tempo": d.tempo,
            "valence": d.valence,
            "track_id": d.track_id
        } for d in data])

        logger.info(f"Data retrieved: {len(df)} rows from Spotify table.")

        # Define number of partitions for Dask DataFrame
        npartitions = 10
        ddf = dd.from_pandas(df, npartitions=npartitions)

        # Select features
        features = ddf[['danceability', 'energy', 'tempo', 'valence']].compute().values
        logger.info(f"Feature array shape: {features.shape}")

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        logger.info("Submitting KMeans clustering tasks to Dask...")

        # Define KMeans function with handling for insufficient data points in a chunk
        def perform_kmeans_on_chunk(chunk):
            kmeans = KMeans(n_clusters=5, random_state=42)
            chunk_computed = chunk.compute()
            if len(chunk_computed) >= kmeans.n_clusters:
                labels = kmeans.fit_predict(chunk_computed)
            else:
                labels = np.zeros(len(chunk_computed))  # Assign zero cluster for small chunks
            return labels

        # Split features into chunks and submit them to Dask
        futures = [client.submit(perform_kmeans_on_chunk, chunk) for chunk in
                   dd.from_array(features_scaled, chunksize=len(features_scaled) // npartitions).to_delayed()]

        # Gather results
        labels = client.gather(futures)

        # Concatenate the results into one array
        labels_concat = pd.concat([pd.Series(l) for l in labels])

        # Convert labels to a Dask array for further processing
        labels_dask = dd.from_array(labels_concat.values, chunksize=npartitions)

        # **Fix the Update Function**: Ensure rows and labels match
        def update_song(df, labels):
            if len(df) != len(labels):
                raise ValueError(f"Mismatch in number of rows: {len(df)} and labels: {len(labels)}")
            df['kmeans'] = labels
            return df

        # Apply the update function
        ddf_updated = ddf.map_partitions(update_song, labels_dask.compute(), meta={'track_id': 'int64', 'kmeans': 'int64'})

        # Convert to Pandas DataFrame for SQL update
        df_updated = ddf_updated.compute()

        # **Optimize the database update**: Update with necessary data
        updates = []  # List to store updates for bulk insert
        for index, row in df_updated.iterrows():
            song = SpotifyData.query.filter_by(track_id=row['track_id']).first()
            if song:
                song.kmeans = row['kmeans']
                updates.append(song)

        if updates:  # Check if there are updates before committing
            db.session.add_all(updates)
            db.session.commit()
            logger.info(f"Successfully updated {len(updates)} records with KMeans labels.")

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error during KMeans clustering or database update: {e}")
        logger.debug(e, exc_info=True)

    finally:
        client.close()
        logger.info("Dask client closed after clustering.")