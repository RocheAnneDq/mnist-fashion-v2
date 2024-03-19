#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml
import time

# Define the Streamlit app
def app():

    if "X_train" not in st.session_state:
        st.session_state.X_train = []
    if "X_test" not in st.session_state:
        st.session_state.X_test = []        
    if "y_train" not in st.session_state:
        st.session_state.y_train = []
    if "y_test" not in st.session_state:
        st.session_state.y_test = []                

    st.subheader('The task: Classify images in the MNIST-Fashion Dataset.')
    text = """The Fashion-MNIST dataset is a collection of images used for training 
    machine learning models, specifically for image classification tasks. 
    \nType of Images: Grayscale images of clothing items (t-shirts, dresses, etc.)
    Image Size: 28x28 pixels
    Number of Images: 70,000 total (60,000 training, 10,000 testing)
    Number of Classes: 10 (each representing a type of clothing)
    \nIt's essentially an update to the classic MNIST dataset which has 
    handwritten digits instead of clothing. Because of its structure and size, 
    it's a popular choice for beginners to experiment with image classification."""
    st.write(text)

    progress_bar = st.progress(0, text="Loading the images, please wait...")

    # Download and load the Fashion MNIST dataset
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True)
    
    # Extract only the specified number of images and labels
    size = 10000
    X = X[:size]
    y = y[:size]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #save objects to the session state
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    st.subheader('First 25 images in the MNIST dataset') 

    # Get the first 25 images and reshape them to 28x28 pixels
    train_images = np.array(X_train)
    train_labels = np.array(y_train)
    images = train_images[:25].reshape(-1, 28, 28)
    # Create a 5x5 grid of subplots
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    # Plot each image on a separate subplot
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(images[i], cmap=plt.cm.binary)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Digit: {train_labels[i]}")
    # Show the plot
    plt.tight_layout()
    st.pyplot(fig)

    # update the progress bar
    for i in range(100):
        # Update progress bar value
        progress_bar.progress(i + 1)
        # Simulate some time-consuming task (e.g., sleep)
        time.sleep(0.01)
    
    # Progress bar reaches 100% after the loop completes
    st.success("Image dataset loading completed!") 

    text = "Go to the Performance page and select a classifier to test its performance."

#run the app
if __name__ == "__main__":
    app()
