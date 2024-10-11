#kmeans.py

import pandas as pd
from sklearn.cluster import KMeans
from app import db
from app.models import SpotifyData
import logging
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

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
        #df = pd.read_sql(db.session.query(SpotifyData.track_id, SpotifyData.danceability, 
        #                                   SpotifyData.energy, SpotifyData.tempo, 
        #                                   SpotifyData.valence).statement, db.session.engine)
        df = pd.read_sql(
            "SELECT track_id, danceability, energy, acousticness, valence FROM Spotify",
            engine  # Use the passed engine here
        )

        if df.empty:
            logger.warning("No data retrieved from Spotify table.")
            return

        logger.info(f"Data retrieved: {len(df)} rows from Spotify table.")

        total_rows = len(df)
        # Scale features
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
            batch['kmeans'] = labels  # Change here to use 'kmeans'

            # Bulk update using SQLAlchemy
            session = db.session
            updates = []

            for index, row in batch.iterrows():
                updates.append({
                    'track_id': row['track_id'],
                    'kmeans': row['kmeans']  # Change here to use 'kmeans'
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

def generate_kmeans_graph(df):
    # Log the initial state of the DataFrame for debugging
    logger.info(f"DataFrame for KMeans graph: {df.head()}")

    if df.empty:
        logger.warning("No KMeans results found for graphing.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(df['danceability'], df['energy'], c=df['kmeans'], cmap='viridis', alpha=0.5)
    plt.title('KMeans Clustering')
    plt.xlabel('Danceability')
    plt.ylabel('Energy')
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
