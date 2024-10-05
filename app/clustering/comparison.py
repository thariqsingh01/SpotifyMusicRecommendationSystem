import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load your pre-trained CNN model
def load_cnn_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def perform_cnn_clustering(data):
    # Example: Extract features from the data
    features = data[['danceability', 'energy']].values  # Adjust based on your feature set

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Split data into training and testing sets
    X_train, X_test = train_test_split(scaled_features, test_size=0.2, random_state=42)

    # Load the pre-trained CNN model
    model = load_cnn_model('path_to_your_model.h5')  # Adjust this path

    # Perform predictions to get cluster assignments
    cnn_predictions = model.predict(X_test)
    cnn_clusters = np.argmax(cnn_predictions, axis=1)  # Assuming your model outputs class probabilities

    # Update the 'cnn' column in the Spotify table
    # Assuming you have access to the original data to match with predictions
    data.loc[data.index.isin(X_test.index), 'cnn'] = cnn_clusters  # Match clusters back to original indices

    return data

def generate_cnn_graph(data):
    # Filter out the data where 'cnn' is not null
    filtered_data = data[data['cnn'].notnull()]

    plt.figure(figsize=(10, 6))

    # Create a scatter plot based on the clusters
    plt.scatter(filtered_data['danceability'], filtered_data['energy'], c=filtered_data['cnn'], cmap='viridis', marker='o')

    plt.title("CNN Clustering Results")
    plt.xlabel("Danceability")
    plt.ylabel("Energy")
    plt.colorbar(label='Cluster')
    
    plt.savefig('static/graphs/cnn_results.png')
    plt.close()
