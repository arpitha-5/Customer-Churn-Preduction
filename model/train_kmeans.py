import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_DIR, "model", "saved_models")
PROCESSED_DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "processed_data.pkl")

def train_segmentation():
    print("Loading processed data for clustering...")
    try:
        data = joblib.load(PROCESSED_DATA_PATH)
        X_train = data["X_train"]
        feature_names = data["feature_names"]
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Training K-Means Clustering Model (3 segments)...")
    # Using 3 clusters to keep it simple and interpretable
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_train)

    # Analyze clusters
    centroids = kmeans.cluster_centers_
    df_centroids = pd.DataFrame(centroids, columns=feature_names)
    
    # We will determine the logic for the segments based on typical churn characteristics
    # E.g., looking at 'tenure' and 'MonthlyCharges' in the scaled data.
    # We will just map the cluster IDs (0, 1, 2) to names dynamically in the backend.

    model_path = os.path.join(MODEL_DIR, "kmeans_model.pkl")
    joblib.dump(kmeans, model_path)
    print(f"K-Means model saved to {model_path}")
    
    print("Done!")

if __name__ == "__main__":
    train_segmentation()
