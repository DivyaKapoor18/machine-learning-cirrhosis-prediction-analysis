#THE ORIGINAL CODE 
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.cluster import KMeans, DBSCAN
# from sklearn.metrics import silhouette_score
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# # Load the dataset
# file_path = './cirrhosis.csv'  # Update path as necessary
# cirrhosis_data = pd.read_csv(file_path)

# # Preprocessing: Drop irrelevant columns and handle missing values
# cirrhosis_data_cleaned = cirrhosis_data.drop(columns=["ID"])  # Drop ID column
# numerical_features = cirrhosis_data_cleaned.select_dtypes(include=["float64", "int64"]).columns
# cirrhosis_data_cleaned[numerical_features] = cirrhosis_data_cleaned[numerical_features].fillna(
#     cirrhosis_data_cleaned[numerical_features].mean()
# )

# # One-hot encode categorical features
# categorical_features = cirrhosis_data_cleaned.select_dtypes(include=["object"]).columns
# encoder = OneHotEncoder(sparse_output=False, drop="first")
# encoded_data = encoder.fit_transform(cirrhosis_data_cleaned[categorical_features])
# encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))

# # Combine processed data
# cirrhosis_preprocessed = pd.concat(
#     [cirrhosis_data_cleaned[numerical_features].reset_index(drop=True), encoded_df.reset_index(drop=True)],
#     axis=1
# )

# # Standardize the features
# scaler = StandardScaler()
# cirrhosis_scaled = scaler.fit_transform(cirrhosis_preprocessed)

# # Apply K-Means clustering
# kmeans = KMeans(n_clusters=3, random_state=42)
# kmeans_labels = kmeans.fit_predict(cirrhosis_scaled)

# # Evaluate K-Means clustering performance
# kmeans_silhouette = silhouette_score(cirrhosis_scaled, kmeans_labels)
# print(f"K-Means Silhouette Score: {kmeans_silhouette:.2f}")

# # Apply DBSCAN clustering
# dbscan = DBSCAN(eps=3.6, min_samples=5)  # Adjust hyperparameters as needed
# dbscan_labels = dbscan.fit_predict(cirrhosis_scaled)

# # Analyze DBSCAN clustering performance
# unique_labels = set(dbscan_labels)
# n_clusters_dbscan = len(unique_labels) - (1 if -1 in unique_labels else 0)
# print(f"DBSCAN: Number of clusters (excluding noise): {n_clusters_dbscan}")

# # Visualize clusters using PCA
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(cirrhosis_scaled)

# plt.figure(figsize=(12, 6))

# # K-Means visualization
# plt.subplot(121)
# plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_labels, cmap="viridis", s=50, alpha=0.7)
# plt.title(f"K-Means Clustering (Silhouette Score: {kmeans_silhouette:.2f})", fontsize=14)
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.colorbar(label="Cluster Label")

# # DBSCAN visualization
# plt.subplot(122)
# plt.scatter(pca_result[:, 0], pca_result[:, 1], c=dbscan_labels, cmap="viridis", s=50, alpha=0.7)
# plt.title(f"DBSCAN Clustering (Clusters: {n_clusters_dbscan})", fontsize=14)
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.colorbar(label="Cluster Label")

# plt.tight_layout()
# plt.show()

# # Cluster Analysis Function
# def analyze_clusters(labels, method_name):
#     print(f"\n{method_name} Clustering Analysis:")
#     cluster_count = len(np.unique(labels)) - (1 if -1 in labels else 0)  # Exclude noise points
#     print("Number of clusters (excluding noise):", cluster_count)
#     df_clustered = cirrhosis_preprocessed.copy()
#     df_clustered['cluster'] = labels

#     for cluster in range(cluster_count):
#         cluster_data = df_clustered[df_clustered['cluster'] == cluster]
#         print(f"\nCluster {cluster} Characteristics:")
#         print("Cluster Size:", len(cluster_data))
#         for col in numerical_features:
#             print(f"Average {col}:", cluster_data[col].mean())

# # Perform cluster analysis
# analyze_clusters(kmeans_labels, "K-Means")
# analyze_clusters(dbscan_labels, "DBSCAN")


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
file_path = '../data/cirrhosis.csv'
cirrhosis_data = pd.read_csv(file_path)

# Preprocessing: Drop irrelevant columns and handle missing values
cirrhosis_data_cleaned = cirrhosis_data.drop(columns=["ID"])
numerical_features = cirrhosis_data_cleaned.select_dtypes(include=["float64", "int64"]).columns
cirrhosis_data_cleaned[numerical_features] = cirrhosis_data_cleaned[numerical_features].fillna(
    cirrhosis_data_cleaned[numerical_features].mean()
)

# One-hot encode categorical features
categorical_features = cirrhosis_data_cleaned.select_dtypes(include=["object"]).columns
encoder = OneHotEncoder(sparse_output=False, drop="first")
encoded_data = encoder.fit_transform(cirrhosis_data_cleaned[categorical_features])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))

# Combine processed data
cirrhosis_preprocessed = pd.concat(
    [cirrhosis_data_cleaned[numerical_features].reset_index(drop=True), encoded_df.reset_index(drop=True)],
    axis=1
)

# Standardize the features
scaler = StandardScaler()
cirrhosis_scaled = scaler.fit_transform(cirrhosis_preprocessed)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(cirrhosis_scaled)

# Evaluate K-Means clustering performance
kmeans_silhouette = silhouette_score(cirrhosis_scaled, kmeans_labels)
print(f"K-Means Silhouette Score: {kmeans_silhouette:.2f}")

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=3.6, min_samples=5)
dbscan_labels = dbscan.fit_predict(cirrhosis_scaled)

# Analyze DBSCAN clustering performance
unique_labels = set(dbscan_labels)
n_clusters_dbscan = len(unique_labels) - (1 if -1 in unique_labels else 0)
print(f"DBSCAN: Number of clusters (excluding noise): {n_clusters_dbscan}")

# Visualize clusters using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(cirrhosis_scaled)

plt.figure(figsize=(12, 6))

# K-Means visualization
plt.subplot(121)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_labels, cmap="viridis", s=50, alpha=0.7)
plt.title(f"K-Means Clustering (Silhouette Score: {kmeans_silhouette:.2f})", fontsize=14)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster Label")

# DBSCAN visualization
plt.subplot(122)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=dbscan_labels, cmap="viridis", s=50, alpha=0.7)
plt.title(f"DBSCAN Clustering (Clusters: {n_clusters_dbscan})", fontsize=14)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster Label")

plt.tight_layout()
plt.show()

# Cluster Analysis Function
def analyze_clusters(labels, method_name):
    print(f"\n{method_name} Clustering Analysis:")
    cluster_count = len(np.unique(labels)) - (1 if -1 in labels else 0)
    print("Number of clusters (excluding noise):", cluster_count)
    df_clustered = cirrhosis_preprocessed.copy()
    df_clustered['cluster'] = labels

    for cluster in range(cluster_count):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster]
        print(f"\nCluster {cluster} Characteristics:")
        print("Cluster Size:", len(cluster_data))
        for col in numerical_features:
            print(f"Average {col}:", cluster_data[col].mean())

# Perform cluster analysis
analyze_clusters(kmeans_labels, "K-Means")
analyze_clusters(dbscan_labels, "DBSCAN")