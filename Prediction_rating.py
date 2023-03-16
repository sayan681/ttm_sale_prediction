import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearnex import patch_sklearn
patch_sklearn()

# Load the dataset
data = pd.read_csv(r'food.csv')

# Extract the features
features = data[['Price', 'Rating']]

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled = pd.DataFrame(features_scaled, columns=features.columns)

# Train the KMeans model
kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, algorithm='lloyd')
kmeans.fit(features_scaled)

# Predict the cluster for a new item
new_item = [[290, 4.8]]  # Price (in Rupees), Rating (out of 5)
new_item_scaled = scaler.transform(new_item)
new_item_scaled = pd.DataFrame(new_item_scaled, columns=features.columns)
cluster = kmeans.predict(new_item_scaled)

# Determine if the new item is likely to have a high number of times sold based on the cluster
if cluster == 1:
    print('The new item is likely to have a high number of sales.')
else:
    print('The new item is not likely to have a high number of sales.')
