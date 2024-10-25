#kmeans.py

import pandas as pd
from sklearn.cluster import KMeans
from app import db
from app.models import SpotifyData
import logging
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_kmeans_clustering(engine, n_clusters=100, batch_size=50000):
    try:
        # Check if KMeans clustering has already been performed
        result = db.session.query(SpotifyData).filter(SpotifyData.kmeans.isnot(None)).count()

        if result > 0:
            logger.info(f"KMeans clustering already performed on {result} records. Skipping clustering.")
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
        # Normalize the features
        scaler = StandardScaler()
        df[['danceability', 'energy', 'acousticness', 'valence']] = scaler.fit_transform(df[['danceability', 'energy', 'acousticness', 'valence']])
        logger.info(f"Data scaled. Sample: {df[['danceability', 'energy', 'acousticness', 'valence']].head()}")

        # Process data in batches
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            batch = df.iloc[start:end]

            # Perform KMeans clustering on the batch
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(batch[['danceability', 'energy', 'acousticness', 'valence']])
            batch['kmeans'] = labels 

            # Bulk update using SQLAlchemy
            session = db.session
            updates = []

            for index, row in batch.iterrows():
                updates.append({
                    'track_id': row['track_id'],
                    'kmeans': row['kmeans']  
                })

            if updates:
                session.bulk_update_mappings(SpotifyData, updates)
                session.commit()
                logger.info(f"Successfully updated {len(updates)} records with KMeans labels from rows {start} to {end}.")
            else:
                logger.warning(f"No updates to apply for rows {start} to {end}.")

    except Exception as e:
        logger.error(f"Error during KMeans Clustering or database update: {e}")
        db.session.rollback()
        logger.debug(e, exc_info=True)

    finally:
        logger.info("Completed KMeans Clustering.")

def generate_kmeans_graph(features, labels):
    # Log the initial state of the DataFrame for debugging
    logger.info(f"DataFrame for KMeans graph: {features.head()}, Labels: {labels.head()}")

    if features.empty or labels.empty:
        logger.warning("No KMeans results found for graphing.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(features['pca1'], features['pca2'], c=labels, cmap='viridis', alpha=0.5)
    plt.title('KMeans Clustering')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(label='Cluster')

    graph_dir = 'app/static/graphs/'
    os.makedirs(graph_dir, exist_ok=True)
    graph_path = os.path.join(graph_dir, 'kmeans_results.png')

    try:
        plt.savefig(graph_path)
        logger.info(f"KMeans graph saved at {graph_path}")
    except Exception as e:
        logger.error(f"Error saving KMeans graph: {e}")
    plt.close()