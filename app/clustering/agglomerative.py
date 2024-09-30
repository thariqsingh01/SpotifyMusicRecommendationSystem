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

# agglomerative.py
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
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

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[['danceability', 'energy', 'tempo', 'valence']])

        # Initialize Agglomerative Clustering
        agglomerative = AgglomerativeClustering(n_clusters=5)
        logger.info("Fitting Agglomerative Clustering model...")

        # Fit the model to the scaled data
        cluster_assignments = agglomerative.fit_predict(scaled_features)
        logger.info("Agglomerative Clustering model fitted successfully.")
        
        # Add cluster labels to the original DataFrame
        df['agglomerative'] = cluster_assignments

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
        logger.error(f"Error during Agglomerative Clustering or database update: {e}")
        db.session.rollback()  # Rollback the session on error
        logger.debug(e, exc_info=True)

    finally:
        logger.info("Completed Agglomerative Clustering.")
