#agglomerative.py

"""
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from app import db

def perform_agglomerative_clustering():
    from app.models import SpotifyData
    # Query all data
    data = SpotifyData.query.all()
    
    # Extract features using selected variables
    features = [[d.danceability, d.energy, d.tempo, d.valence] for d in data]
    
    # Perform Agglomerative clustering
    agglomerative = AgglomerativeClustering(n_clusters=10)  # Choose number of clusters
    labels = agglomerative.fit_predict(features)

    # Save cluster labels to the database
    for i, song in enumerate(data):
        song.agglomerative = labels[i]
        db.session.add(song)
    db.session.commit()
"""


import faiss
import torch
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from scipy.cluster.hierarchy import linkage, fcluster
from dask import dataframe as dd
from app import db
from app.models import SpotifyData
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_agglomerative_clustering(uri, engine, n_clusters=5, use_gpu=True):
    try:
        # Retrieve data from Spotify table using Pandas
        query = "SELECT danceability, energy, tempo, valence, track_id FROM Spotify"
        df = pd.read_sql(query, engine)

        if df.empty:
            logger.warning("No data retrieved from Spotify table.")
            return

        logger.info(f"Data retrieved: {len(df)} rows from Spotify table.")
        
        # Use Dask to handle large datasets
        ddf = dd.from_pandas(df, npartitions=10)

        # Convert to PyTorch tensor (use GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        data = torch.tensor(ddf[['danceability', 'energy', 'tempo', 'valence']].compute().values, dtype=torch.float32).to(device)

        # Use FAISS for nearest neighbors search
        index = faiss.IndexFlatL2(data.shape[1])
        index.add(data.cpu().numpy())
        distances, _ = index.search(data.cpu().numpy(), len(data))
        
        logger.info("FAISS nearest neighbors search completed.")

        # Use Dask to compute pairwise distances and linkage
        pairwise_dists = delayed(pairwise_distances)(data.cpu().numpy())
        Z = delayed(linkage)(pairwise_dists, method='ward')
        cluster_assignments = delayed(fcluster)(Z, n_clusters, criterion='maxclust')

        # Compute Dask delayed values
        cluster_assignments = cluster_assignments.compute()

        df['agglomerative'] = cluster_assignments

        # Bulk update using SQLAlchemy
        session = db.session
        updates = []

        for index, row in df.iterrows():
            updates.append({
                'track_id': row['track_id'],
                'agglomerative': row['agglomerative']
            })

        if updates:
            session.bulk_update_mappings(SpotifyData, updates)
            session.commit()
            logger.info(f"Successfully updated {len(updates)} records with Agglomerative Clustering labels.")

    except Exception as e:
        logger.error(f"Error during Agglomerative Clustering or database update: {e}")
        db.session.rollback()
        logger.debug(e, exc_info=True)

    finally:
        logger.info("Completed Agglomerative Clustering.")
