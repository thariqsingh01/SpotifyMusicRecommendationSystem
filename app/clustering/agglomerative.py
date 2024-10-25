#agglomerative.py

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from app import db
from app.models import SpotifyData
import logging
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_agglomerative_clustering(engine, n_clusters=100, batch_size=50000):
    try:
        # Check if Agglomerative Clustering has already been performed
        result = db.session.query(SpotifyData).filter(SpotifyData.agglomerative.isnot(None)).count()

        if result > 0:
            logger.info(f"Agglomerative clustering already performed on {result} records. Skipping clustering.")
            return

        # Retrieve data from Spotify table using Pandas
        df = pd.read_sql(
            "SELECT track_id, danceability, energy, acousticness, valence FROM Spotify",
            engine 
        )


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
            labels = agglomerative.fit_predict(batch[['danceability', 'energy', 'acousticness', 'valence']])
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

def generate_agglomerative_graph(features, labels):
    logger.info(f"DataFrame for Agglomerative graph: {features.head()}, Labels: {labels.head()}")

    if features.empty or labels.empty:
        logger.warning("No Agglomerative results found for graphing.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(features['pca1'], features['pca2'], c=labels, cmap='cividis', alpha=0.5)
    plt.title('Agglomerative Clustering')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Cluster')

    graph_dir = 'app/static/graphs/'
    os.makedirs(graph_dir, exist_ok=True)
    graph_path = os.path.join(graph_dir, 'agglomerative_results.png')

    try:
        plt.savefig(graph_path)
        logger.info(f"Agglomerative graph saved at {graph_path}")
    except Exception as e:
        logger.error(f"Error saving Agglomerative graph: {e}")
    plt.close()