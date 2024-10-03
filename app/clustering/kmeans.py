#kmeans.py

import faiss
import pandas as pd
import torch
from dask import dataframe as dd
from app import db
from app.models import SpotifyData
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_kmeans_clustering(uri, engine, n_clusters=5, use_gpu=True):
    try:
        # Check if clustering has already been performed by querying the `kmeans` column
        check_query = "SELECT COUNT(*) FROM Spotify WHERE kmeans IS NOT NULL"
        result = engine.execute(check_query).scalar()

        if result > 0:
            logger.info(f"KMeans clustering already performed on {result} records. Skipping clustering.")
            return

        # Retrieve data from Spotify table using Pandas
        query = "SELECT danceability, energy, tempo, valence, track_id FROM Spotify"
        df = pd.read_sql(query, engine)

        if df.empty:
            logger.warning("No data retrieved from Spotify table.")
            return

        logger.info(f"Data retrieved: {len(df)} rows from Spotify table.")

        # Use Dask to handle large datasets
        ddf = dd.from_pandas(df, npartitions=10)  # Adjust partitions as needed
        
        # Convert to PyTorch tensor (use GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        data = torch.tensor(ddf[['danceability', 'energy', 'tempo', 'valence']].compute().values, dtype=torch.float32).to(device)

        # Use FAISS's GPU version if available, else fallback to CPU
        if device.type == 'cuda':
            res = faiss.StandardGpuResources()
            kmeans = faiss.Kmeans(d=data.shape[1], k=n_clusters, gpu=True)
        else:
            kmeans = faiss.Kmeans(d=data.shape[1], k=n_clusters, gpu=False)

        logger.info("Fitting Faiss KMeans model...")

        # Move the data back to CPU for FAISS compatibility
        kmeans.train(data.cpu().numpy())
        logger.info("Faiss KMeans model fitted successfully.")

        # Get cluster assignments
        _, cluster_assignments = kmeans.index.search(data.cpu().numpy(), 1)
        
        # Add cluster labels to the original DataFrame
        df['kmeans'] = cluster_assignments.flatten()

        # Bulk update using SQLAlchemy
        session = db.session
        updates = []

        for index, row in df.iterrows():
            updates.append({
                'track_id': row['track_id'],
                'kmeans': row['kmeans']
            })

        if updates:
            session.bulk_update_mappings(SpotifyData, updates)
            session.commit()
            logger.info(f"Successfully updated {len(updates)} records with KMeans labels.")

    except Exception as e:
        logger.error(f"Error during Faiss KMeans clustering or database update: {e}")
        db.session.rollback()
        logger.debug(e, exc_info=True)

    finally:
        logger.info("Completed Faiss KMeans clustering.")
