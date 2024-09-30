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


import faiss
import torch
import pandas as pd
from dask import delayed, compute
from app import db
from app.models import SpotifyData
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_dbscan_clustering(uri, engine, eps=0.5, min_samples=5, use_gpu=True):
    try:
        # Retrieve data from Spotify table using Pandas
        query = "SELECT danceability, energy, tempo, valence, track_id FROM Spotify"
        df = pd.read_sql(query, engine)

        if df.empty:
            logger.warning("No data retrieved from Spotify table.")
            return

        logger.info(f"Data retrieved: {len(df)} rows from Spotify table.")
        
        # Use Dask to handle large datasets
        ddf = dd.from_pandas(df, npartitions=10)  # Adjust partitions

        # Convert to PyTorch tensor (use GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        data = torch.tensor(ddf[['danceability', 'energy', 'tempo', 'valence']].compute().values, dtype=torch.float32).to(device)

        # FAISS for nearest neighbors search
        index = faiss.IndexFlatL2(data.shape[1])
        index.add(data.cpu().numpy())
        distances, neighbors = index.search(data.cpu().numpy(), min_samples)
        
        logger.info("FAISS nearest neighbors search completed.")

        # Parallel DBSCAN computation with Dask
        def dbscan_point(idx, point_neighbors):
            density = (point_neighbors < eps).sum()
            return 1 if density >= min_samples else -1  # 1: core point, -1: noise

        cluster_labels = [delayed(dbscan_point)(i, distances[i]) for i in range(len(data))]
        cluster_labels = compute(*cluster_labels)

        df['dbscan'] = cluster_labels

        # Bulk update using SQLAlchemy
        session = db.session
        updates = []

        for index, row in df.iterrows():
            updates.append({
                'track_id': row['track_id'],
                'dbscan': row['dbscan']
            })

        if updates:
            session.bulk_update_mappings(SpotifyData, updates)
            session.commit()
            logger.info(f"Successfully updated {len(updates)} records with DBSCAN labels.")

    except Exception as e:
        logger.error(f"Error during DBSCAN clustering or database update: {e}")
        db.session.rollback()
        logger.debug(e, exc_info=True)

    finally:
        logger.info("Completed DBSCAN Clustering.")
