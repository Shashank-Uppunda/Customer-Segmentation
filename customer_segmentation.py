# Customer Segmentation using K-Means Clustering

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("mall_customers.csv")

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Business labels
business_labels = {
    0: "Middle-income avg spenders",
    1: "High-income high spenders",
    2: "Low-income high spenders",
    3: "High-income low spenders",
    4: "Low-income low spenders"
}

# Get cluster centers in original scale
centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)

# Plot clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
            c=df['Cluster'], cmap='viridis', s=50, alpha=0.6, edgecolors='w')

# Plot centroids
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')

# Annotate centroids with business labels using a simple loop
i = 0
for point in centers:
    x = point[0]
    y = point[1]
    label = business_labels[i]
    plt.text(x, y, label, fontsize=9, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='white', bbox=dict(facecolor='blue', alpha=0.7, boxstyle='round,pad=0.3'))
    i += 1  # increment index

# Annual Income is an estimate derived from customers' demographic data such as age, location, and spending behavior.
plt.xlabel('Estimated Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation with K-Means Clusters and Centroids')
plt.legend()
plt.show()