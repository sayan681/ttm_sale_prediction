import pandas as pd
from sklearn.cluster import KMeans
from sklearnex import patch_sklearn
from sklearn.preprocessing import MinMaxScaler

patch_sklearn()

food_df = pd.read_csv(r'F:\Programming\oneAPI-Scikitlearn-main\food_data.csv')

# Drop the Restaurant Name and Dish Name columns
food_df.drop(['Restaurant Name', 'Dish Name'], axis=1, inplace=True)

# Scale the Price and Cost columns using MinMaxScaler
scaler = MinMaxScaler()
food_df[['Price of the food', 'Cost of making the food']] = scaler.fit_transform(food_df[['Price of the food', 'Cost of making the food']])

# Define the features and target variable
X = food_df.drop('Number of dishes sold', axis=1)
y = food_df['Number of dishes sold']

# Use the Intel optimized version of KMeans
kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0,
                random_state=None, copy_x=True)

# Fit the KMeans model to the food data
kmeans.fit(X)

# Predict the clusters for the food data
y_pred = kmeans.predict(X)

# Add the predicted cluster labels to the food dataset
food_df['Cluster'] = y_pred

# Compute the mean number of dishes sold for each cluster
mean_sales = food_df.groupby('Cluster')['Number of dishes sold'].mean()

# Define a function to predict whether a new dish will have high sales
def predict_high_sales(price, cost):
    # Scale the price and cost using MinMaxScaler
    price_scaled, cost_scaled = scaler.transform([[price, cost]])[0]
    
    # Predict the cluster for the new dish
    cluster = kmeans.predict([[price_scaled, cost_scaled]])[0]
    
    # Return True if the mean number of dishes sold for the predicted cluster is high, False otherwise
    return mean_sales[cluster] > y.mean()

# Example usage:
result = predict_high_sales(8,2)
print(result) # True or False
