# comparison.py

from flask import Blueprint, render_template, current_app
from app import db
from app.models import SpotifyData
import pandas as pd
import logging

bp = Blueprint('comparison', __name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_cluster_data():
    """
    Fetches cluster data from the SpotifyData model for comparison.
    Returns: DataFrame: A DataFrame containing cluster group counts and other characteristics.
    """
    with current_app.app_context():
        # Execute a query to get cluster group counts for KMeans, DBSCAN, and Agglomerative
        cluster_query = """
            SELECT
                kmeans AS cluster_group,
                COUNT(*) AS count,
                'KMeans' AS algorithm
            FROM SpotifyData
            WHERE kmeans IS NOT NULL
            GROUP BY kmeans

            UNION ALL

            SELECT
                dbscan AS cluster_group,
                COUNT(*) AS count,
                'DBSCAN' AS algorithm
            FROM SpotifyData
            WHERE dbscan IS NOT NULL
            GROUP BY dbscan

            UNION ALL

            SELECT
                agglomerative AS cluster_group,
                COUNT(*) AS count,
                'Agglomerative' AS algorithm
            FROM SpotifyData
            WHERE agglomerative IS NOT NULL
            GROUP BY agglomerative

            ORDER BY algorithm, cluster_group;
        """
        # Execute the query and convert the results to a DataFrame
        combined_results = pd.read_sql_query(cluster_query, db.engine)

        logger.info("Cluster data fetched successfully.")
        return combined_results

def fetch_recommendations():
    """
    Fetches sample recommendations based on user choices for comparison.
    Returns: list: A list of dictionaries containing recommendations for different clustering methods.
    """
    sample_recommendations = {
        "KMeans": [
            {'track_name': 'Song A', 'artist_name': 'Artist A', 'genre': 'Pop', 'year': 2022},
            {'track_name': 'Song B', 'artist_name': 'Artist B', 'genre': 'Rock', 'year': 2021},
        ],
        "DBSCAN": [
            {'track_name': 'Song C', 'artist_name': 'Artist C', 'genre': 'Jazz', 'year': 2020},
            {'track_name': 'Song D', 'artist_name': 'Artist D', 'genre': 'Hip Hop', 'year': 2023},
        ],
        "Agglomerative": [
            {'track_name': 'Song E', 'artist_name': 'Artist E', 'genre': 'Classical', 'year': 2019},
            {'track_name': 'Song F', 'artist_name': 'Artist F', 'genre': 'Indie', 'year': 2018},
        ]
    }
    
    logger.info("Sample recommendations fetched successfully.")
    return sample_recommendations

@bp.route('/comparison', methods=['GET'])
def comparison():
    """
    Renders the comparison page with cluster data and recommendations.
    Returns: HTML: The rendered comparison page.
    """
    try:
        cluster_data = fetch_cluster_data()
        recommendations = fetch_recommendations()

        return render_template('comparison.html', cluster_data=cluster_data, recommendations=recommendations)

    except Exception as e:
        logger.error(f"Error fetching comparison data: {e}")
        return render_template('comparison.html', error="An error occurred while fetching data.")
