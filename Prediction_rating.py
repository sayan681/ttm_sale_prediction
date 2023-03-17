import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearnex import patch_sklearn
import plotly.express as px

patch_sklearn()

# Set the title and description of the Streamlit app
st.set_page_config(page_title="Food Sales Prediction",
                   page_icon=":pizza:", layout="wide")
st.title("Food Sales Prediction")

st.markdown("*This app predicts the number of times a new food item is likely to be sold based on its price and rating, using a more advanced and optimized KMeans clustering algorithm by Intel(R) Extension for Scikit-learn enabled (https://github.com/intel/scikit-learn-intelex)*")

# Load the dataset
data = pd.read_csv(r'food.csv')

# Extract the features
features = data[['Price', 'Rating']]

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled = pd.DataFrame(features_scaled, columns=features.columns)

# Train the KMeans model
kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10,
                max_iter=300, algorithm='lloyd')
kmeans.fit(features_scaled)

# Set up the sidebar and plot area
st.sidebar.header('New Item Details')
new_price = st.sidebar.number_input('Price (in Rupees)', value=290)
new_rating = st.sidebar.slider(
    'Rating (out of 5)', min_value=0.0, max_value=5.0, step=0.1, value=3.8)
new_item = [[new_price, new_rating]]
new_item_scaled = scaler.transform(new_item)
new_item_scaled = pd.DataFrame(new_item_scaled, columns=features.columns)
cluster = kmeans.predict(new_item_scaled)

# Determine if the new item is likely to have a high number of times sold based on the cluster
if cluster == 1:
    st.subheader('The new item is likely to have a high number of sales.')
else:
    st.subheader('The new item is not likely to have a high number of sales.')

# Plot the data with clusters
fig = px.scatter(features_scaled, x='Price', y='Rating', color=kmeans.labels_)

st.plotly_chart(fig)


st.markdown('*This is a prototype ML model to be used in trythemenu.com and currently running on a synthetic dataset developed by Sayan Malakar*')
