#dbscan.py

from sklearn.cluster import DBSCAN
import pandas as pd
from app import db
from app.models import SpotifyData
import logging
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_dbscan_clustering(uri, engine, eps=0.05, min_samples=1000, batch_size=10000):
    try:
        # Check if DBSCAN clustering has already been performed
        check_query = "SELECT COUNT(*) FROM Spotify WHERE dbscan IS NOT NULL"
        result = engine.execute(check_query).scalar()

        if result > 0:
            logger.info(f"DBSCAN clustering already performed on {result} records. Skipping clustering.")
            return

        # Retrieve data from Spotify table
        query = "SELECT danceability, energy, tempo, valence, track_id FROM Spotify"
        df = pd.read_sql(query, engine)

        if df.empty:
            logger.warning("No data retrieved from Spotify table.")
            return

        logger.info(f"Data retrieved: {len(df)} rows from Spotify table.")

        # Scale features
        scaler = StandardScaler()
        df[['danceability', 'energy', 'tempo', 'valence']] = scaler.fit_transform(df[['danceability', 'energy', 'tempo', 'valence']])
        logger.info(f"Data scaled. Sample: {df[['danceability', 'energy', 'tempo', 'valence']].head()}")

        total_rows = len(df)
        # Process data in batches
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            batch = df.iloc[start:end]

            # Perform DBSCAN clustering on the batch
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = dbscan.fit_predict(batch[['danceability', 'energy', 'tempo', 'valence']])
            batch['dbscan'] = labels

            # Log the number of noise points and core points
            logger.info(f"Number of noise points: {sum(labels == -1)}")
            logger.info(f"Number of core points in cluster 0: {sum(labels == 0)}")

            # Bulk update using SQLAlchemy
            session = db.session
            updates = []

            for index, row in batch.iterrows():
                if row['dbscan'] != -1:  # Ignore noise (-1) if necessary
                    updates.append({
                        'track_id': row['track_id'],
                        'dbscan': row['dbscan']
                    })

            if updates:
                session.bulk_update_mappings(SpotifyData, updates)
                session.commit()
                logger.info(f"Successfully updated {len(updates)} records with DBSCAN labels from rows {start} to {end}.")
            else:
                logger.warning(f"No updates to apply for rows {start} to {end}.")

    except Exception as e:
        logger.error(f"Error during DBSCAN Clustering or database update: {e}")
        db.session.rollback()
        logger.debug(e, exc_info=True)

    finally:
        logger.info("Completed DBSCAN Clustering.")
