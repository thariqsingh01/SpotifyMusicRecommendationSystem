#agglomerative.py

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from app import db
from app.models import SpotifyData
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_agglomerative_clustering(uri, engine, n_clusters=5, batch_size=10000):
    try:
        # Check if Agglomerative Clustering has already been performed
        check_query = "SELECT COUNT(*) FROM Spotify WHERE agglomerative IS NOT NULL"
        result = engine.execute(check_query).scalar()

        if result > 0:
            logger.info(f"Agglomerative clustering already performed on {result} records. Skipping clustering.")
            return

        # Retrieve data from Spotify table using Pandas
        query = "SELECT danceability, energy, tempo, valence, track_id FROM Spotify"
        df = pd.read_sql(query, engine)

        if df.empty:
            logger.warning("No data retrieved from Spotify table.")
            return

        logger.info(f"Data retrieved: {len(df)} rows from Spotify table.")

        total_rows = len(df)
        # Process data in batches
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            batch = df.iloc[start:end]

            # Perform Agglomerative clustering on the batch
            agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
            labels = agglomerative.fit_predict(batch[['danceability', 'energy', 'tempo', 'valence']])
            batch['agglomerative'] = labels

            # Bulk update using SQLAlchemy
            session = db.session
            updates = []

            for index, row in batch.iterrows():
                updates.append({
                    'track_id': row['track_id'],
                    'agglomerative': row['agglomerative']
                })

            if updates:
                session.bulk_update_mappings(SpotifyData, updates)
                session.commit()
                logger.info(f"Successfully updated {len(updates)} records with Agglomerative Clustering labels from rows {start} to {end}.")
            else:
                logger.warning(f"No updates to apply for rows {start} to {end}.")

    except Exception as e:
        logger.error(f"Error during Agglomerative Clustering or database update: {e}")
        db.session.rollback()
        logger.debug(e, exc_info=True)

    finally:
        logger.info("Completed Agglomerative Clustering.")
