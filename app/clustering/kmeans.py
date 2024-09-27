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




import logging
import pandas as pd
from sklearn.cluster import KMeans
from dask.distributed import Client
from sklearn.preprocessing import StandardScaler
from app import db
from app.models import SpotifyData
import dask.dataframe as dd
from sqlalchemy import create_engine

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_kmeans_clustering(uri, engine):
    client = Client()
    logger.info("Dask client started for parallel processing.")

    try:
        # Retrieve data from Spotify table using Pandas
        query = "SELECT danceability, energy, tempo, valence, track_id FROM Spotify"
        df = pd.read_sql(query, engine)

        if df.empty:
            logger.warning("No data retrieved from Spotify table.")
            return

        logger.info(f"Data retrieved: {len(df)} rows from Spotify table.")

        # Convert Pandas DataFrame to Dask DataFrame
        ddf = dd.from_pandas(df, npartitions=10)  # Adjust number of partitions as needed

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(ddf[['danceability', 'energy', 'tempo', 'valence']].compute())

        # Initialize KMeans
        kmeans = KMeans(n_clusters=5, random_state=42)

        logger.info("Fitting KMeans model...")
        # Fit the model to the scaled features
        labels = kmeans.fit_predict(features_scaled)

        logger.info("KMeans model fitted successfully.")

        # Add labels to DataFrame
        df['kmeans'] = labels

        # Optimize the database update: Update with necessary data
        updates = []  # List to store updates for bulk insert
        for index, row in df.iterrows():
            song = SpotifyData.query.filter_by(track_id=row['track_id']).first()
            if song:
                song.kmeans = row['kmeans']
                updates.append(song)

        if updates:  # Check if there are updates before committing
            db.session.add_all(updates)
            db.session.commit()
            logger.info(f"Successfully updated {len(updates)} records with KMeans labels.")

    except Exception as e:
        db.session.rollback()  # Rollback the session on error
        logger.error(f"Error during KMeans clustering or database update: {e}")
        logger.debug(e, exc_info=True)

    finally:
        client.close()  # Close Dask client
        logger.info("Dask client closed after clustering.")
"""

import h2o
from h2o.estimators import H2OKMeansEstimator
import pandas as pd
from app import db
from app.models import SpotifyData
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_kmeans_clustering(uri, engine):
    h2o.init()  # Initialize H2O cluster
    logger.info("H2O cluster started.")

    try:
        # Retrieve data from Spotify table using Pandas
        query = "SELECT danceability, energy, tempo, valence, track_id FROM Spotify"
        df = pd.read_sql(query, engine)

        if df.empty:
            logger.warning("No data retrieved from Spotify table.")
            return

        logger.info(f"Data retrieved: {len(df)} rows from Spotify table.")

        # Convert Pandas DataFrame to H2OFrame
        h2o_df = h2o.H2OFrame(df)

        # Initialize KMeans
        kmeans = H2OKMeansEstimator(k=5, seed=42)
        logger.info("Fitting H2O KMeans model...")

        # Fit the model to the data
        kmeans.train(x=['danceability', 'energy', 'tempo', 'valence'], training_frame=h2o_df)

        logger.info("H2O KMeans model fitted successfully.")
        
        # Get cluster assignments
        cluster_assignments = kmeans.predict(h2o_df)
        
        # Add cluster labels to the original DataFrame
        df['kmeans'] = cluster_assignments['predict'].as_data_frame().values.flatten()

        # Bulk update using SQLAlchemy
        session = db.session
        updates = []

        for index, row in df.iterrows():
            updates.append({
                'track_id': row['track_id'],
                'kmeans': row['kmeans']
            })

        # Bulk insert with SQLAlchemy
        if updates:
            session.bulk_update_mappings(SpotifyData, updates)
            session.commit()
            logger.info(f"Successfully updated {len(updates)} records with KMeans labels.")

    except Exception as e:
        logger.error(f"Error during H2O KMeans clustering or database update: {e}")
        db.session.rollback()  # Rollback the session on error
        logger.debug(e, exc_info=True)

    finally:
        h2o.shutdown()  # Shutdown H2O cluster
        logger.info("H2O cluster shut down after clustering.")