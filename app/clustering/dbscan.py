#dbscan.py

from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from app import db
from app.models import SpotifyData
import logging
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_dbscan_clustering(engine, eps=1.5, min_samples=5, batch_size=100000):
    try:
        df = pd.read_sql(
            "SELECT track_id, danceability, energy, tempo, valence FROM Spotify",
            engine
        )

        if df.empty:
            logger.warning("No data retrieved from Spotify table.")
            return

        logger.info(f"Data retrieved: {len(df)} rows from Spotify table.")

        # Scale features
        scaler = StandardScaler()
        df[['danceability', 'energy', 'tempo', 'valence']] = scaler.fit_transform(df[['danceability', 'energy', 'tempo', 'valence']])
        logger.info(f"Data scaled. Sample: {df[['danceability', 'energy', 'tempo', 'valence']].head()}")

        total_rows = len(df)
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            batch = df.iloc[start:end].copy()

            # Perform DBSCAN clustering on the batch
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = dbscan.fit_predict(batch[['danceability', 'energy', 'tempo', 'valence']])
            batch.loc[:, 'dbscan'] = labels

            # Log cluster distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            logger.info(f"DBSCAN cluster distribution: {dict(zip(unique_labels, counts))}")
            
            # Log the number of noise points
            logger.info(f"Number of noise points: {sum(labels == -1)}")

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


def generate_dbscan_graph(df):
    # Log the initial state of the DataFrame for debugging
    logger.info(f"DataFrame for DBSCAN graph: {df.head()}")

    if df.empty:
        logger.warning("No DBSCAN results found for graphing.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(df['danceability'], df['energy'], c=df['dbscan'], cmap='plasma', alpha=0.5)
    plt.title('DBSCAN Clustering')
    plt.xlabel('Danceability')
    plt.ylabel('Energy')
    plt.colorbar(label='Cluster')

    graph_dir = 'app/static/graphs/'
    os.makedirs(graph_dir, exist_ok=True)
    graph_path = os.path.join(graph_dir, 'dbscan_results.png')

    try:
        plt.savefig(graph_path)
        logger.info(f"DBSCAN graph saved at {graph_path}")
    except Exception as e:
        logger.error(f"Error saving DBSCAN graph: {e}")
    plt.close()
