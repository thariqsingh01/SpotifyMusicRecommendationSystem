import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load your pre-trained CNN model
def load_cnn_model(model_path):
    logger.info(f"Loading CNN model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    logger.info("CNN model loaded successfully.")
    return model

def perform_cnn_clustering(data):
    logger.info("Starting CNN clustering process.")

    # Example: Extract features from the data
    features = data[['danceability', 'energy']].values  # Adjust based on your feature set
    logger.debug(f"Extracted features: {features[:5]}")  # Log first 5 rows of features for debugging

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    logger.info("Features scaled successfully.")

    # Split data into training and testing sets
    X_train, X_test = train_test_split(scaled_features, test_size=0.2, random_state=42)
    logger.info("Data split into training and testing sets.")

    # Load the pre-trained CNN model
    model = load_cnn_model('path_to_your_model.h5')  # Adjust this path

    # Perform predictions to get cluster assignments
    cnn_predictions = model.predict(X_test)
    cnn_clusters = np.argmax(cnn_predictions, axis=1)  # Assuming your model outputs class probabilities
    logger.info(f"CNN predictions made, cluster assignments: {cnn_clusters[:5]}")  # Log first 5 predictions

    # Update the 'cnn' column in the Spotify table
    # Assuming you have access to the original data to match with predictions
    data.loc[data.index.isin(X_test.index), 'cnn'] = cnn_clusters  # Match clusters back to original indices
    logger.info("Updated 'cnn' column in the original data.")

    return data

def generate_cnn_graph(data):
    logger.info("Generating CNN clustering graph.")

    # Filter out the data where 'cnn' is not null
    filtered_data = data[data['cnn'].notnull()]
    logger.info(f"Filtered data for graph: {filtered_data.shape[0]} rows with valid 'cnn' values.")

    plt.figure(figsize=(10, 6))

    # Create a scatter plot based on the clusters
    plt.scatter(filtered_data['danceability'], filtered_data['energy'], c=filtered_data['cnn'], cmap='viridis', marker='o')

    plt.title("CNN Clustering Results")
    plt.xlabel("Danceability")
    plt.ylabel("Energy")
    plt.colorbar(label='Cluster')

    # Create the directory if it doesn't exist
    graphs_dir = 'app/static/graphs/'
    os.makedirs(graphs_dir, exist_ok=True)

    plt.savefig(os.path.join(graphs_dir, 'cnn_results.png'))
    logger.info(f"CNN clustering graph saved at: {os.path.join(graphs_dir, 'cnn_results.png')}")
    plt.close()
