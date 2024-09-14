#kmeans.py

#import pandas as pd
#from sklearn.cluster import KMeans
#from app import db

#def perform_kmeans_clustering():
#    from app.models import SpotifyData
#    # Query all data (make sure to call the method with parentheses)
#    data = SpotifyData.query.all()

#    # Extract features using selected variables
#    features = [[d.danceability, d.energy, d.tempo, d.valence] for d in data]
    
    # Perform KMeans clustering
#    kmeans = KMeans(n_clusters=10)  # Choose number of clusters
#    labels = kmeans.fit_predict(features)

    # Save cluster labels to the database
#    for i, song in enumerate(data):
#        song.kmeans = labels[i]
#        db.session.add(song)
#    db.session.commit()


import pandas as pd
from sklearn.cluster import KMeans
from app import db

def perform_kmeans_clustering():
    from app.models import SpotifyData
    # Query all data
    data = SpotifyData.query.all()
    
    # Debugging: Check if data is returned correctly
    if not data:
        print("No data retrieved from SpotifyData table.")
        return

    # Debugging: Print the first few records to inspect
    print("First record data:", data[0] if data else "No data")

    # Extract features using selected variables
    try:
        features = [[d.danceability, d.energy, d.tempo, d.valence] for d in data]
    except AttributeError as e:
        print(f"Error accessing attributes: {e}")
        return

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=10)  # Choose number of clusters
    labels = kmeans.fit_predict(features)

    # Save cluster labels to the database
    for i, song in enumerate(data):
        song.kmeans = labels[i]
        db.session.add(song)
    db.session.commit()


