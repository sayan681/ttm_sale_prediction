This repository contains a prediction model that predicts the number of times a new food item is likely to be sold based on its price and rating. The model uses a more advanced and optimized KMeans clustering algorithm by Intel(R) Extension for Scikit-learn enabled (https://github.com/intel/scikit-learn-intelex). This extension accelerates the performance of scikit-learn algorithms on Intel(R) processors and compatible architectures.

To get started with the model, you need to have Conda terminal and the code file with the required dataset. The Conda terminal and the code file with dataset must be in the same directory. In Conda terminal, you just have to write the following command to start the prediction model:

- streamlit run Prediction_rating.py

The model will display a user interface where you can enter the price and rating of a new food item and see its predicted sales. You can also see the visualization of the clusters formed by the KMeans algorithm.
