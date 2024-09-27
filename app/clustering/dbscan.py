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


import h2o
from h2o.estimators import H2ODBSCANEstimator
import pandas as pd
from app import db
from app.models import SpotifyData
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_h2o_dbscan_clustering(uri, engine):
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

        # Initialize DBSCAN
        dbscan = H2ODBSCANEstimator(epsilon=0.5, min_points=5)
        logger.info("Fitting H2O DBSCAN model...")

        # Fit the model to the data
        dbscan.train(x=['danceability', 'energy', 'tempo', 'valence'], training_frame=h2o_df)

        logger.info("H2O DBSCAN model fitted successfully.")

        # Get cluster assignments
        cluster_assignments = dbscan.predict(h2o_df)

        # Add cluster labels to the original DataFrame
        df['dbscan'] = cluster_assignments['predict'].as_data_frame().values.flatten()

        # Bulk update using SQLAlchemy
        session = db.session
        updates = []

        for index, row in df.iterrows():
            updates.append({
                'track_id': row['track_id'],
                'dbscan': row['dbscan']
            })

        # Bulk insert with SQLAlchemy
        if updates:
            session.bulk_update_mappings(SpotifyData, updates)
            session.commit()
            logger.info(f"Successfully updated {len(updates)} records with DBSCAN labels.")

    except Exception as e:
        logger.error(f"Error during H2O DBSCAN clustering or database update: {e}")
        db.session.rollback()  # Rollback the session on error
        logger.debug(e, exc_info=True)

    finally:
        h2o.shutdown()  # Shutdown H2O cluster
        logger.info("H2O cluster shut down after clustering.")

