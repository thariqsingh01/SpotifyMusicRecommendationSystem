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



import cudf
from cuml.cluster import AgglomerativeClustering as cuAgglomerativeClustering
from cuml.preprocessing import StandardScaler
import pandas as pd
from app import db
from app.models import SpotifyData
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def perform_agglomerative_clustering(uri, engine):
    try:
        # Retrieve data from Spotify table using Pandas
        query = "SELECT danceability, energy, tempo, valence, track_id FROM Spotify"
        df = pd.read_sql(query, engine)

        if df.empty:
            logger.warning("No data retrieved from Spotify table.")
            return

        logger.info(f"Data retrieved: {len(df)} rows from Spotify table.")

        # Convert Pandas DataFrame to cuDF DataFrame
        cu_df = cudf.DataFrame.from_records(df)

        # Scale features (optional but recommended for clustering)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(cu_df[['danceability', 'energy', 'tempo', 'valence']])

        # Initialize cuML Agglomerative Clustering
        agglomerative = cuAgglomerativeClustering(n_clusters=5)  # Set the desired number of clusters
        logger.info("Fitting cuML Agglomerative Clustering model...")

        # Fit the model to the scaled data
        agglomerative.fit(scaled_features)
        logger.info("cuML Agglomerative Clustering model fitted successfully.")
        
        # Get cluster assignments
        cluster_assignments = agglomerative.labels_

        # Add cluster labels to the original DataFrame
        df['agglomerative'] = cluster_assignments.to_array()

        # Bulk update using SQLAlchemy
        session = db.session
        updates = []

        for index, row in df.iterrows():
            updates.append({
                'track_id': row['track_id'],
                'agglomerative': row['agglomerative']
            })

        # Bulk insert with SQLAlchemy
        if updates:
            session.bulk_update_mappings(SpotifyData, updates)
            session.commit()
            logger.info(f"Successfully updated {len(updates)} records with Agglomerative Clustering labels.")

    except Exception as e:
        logger.error(f"Error during cuML Agglomerative Clustering or database update: {e}")
        db.session.rollback()  # Rollback the session on error
        logger.debug(e, exc_info=True)

    finally:
        logger.info("Completed cuML Agglomerative Clustering.")
