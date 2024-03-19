#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():

    st.sidebar.subheader('Select the classifier')

    # Create the selecton of classifier
    clf = KNeighborsClassifier(n_neighbors=5)
    selected_model = 0

    options = ['K Nearest Neighbor', 'Support Vector Machine', 'Naive Bayes']
    selected_option = st.sidebar.selectbox('Select the classifier', options)
    if selected_option =='Support Vector Machine':
        clf = SVC(kernel='linear')
        selected_model = 1
    elif selected_option=='Naive Bayes':        
        clf = GaussianNB()
        selected_model = 2
    else:
        clf = KNeighborsClassifier(n_neighbors=5)
        selected_model = 0

    if st.button("Begin Test"):

        st.write('List of object class names in Fashion-MNIST')
        # Get the class names from the dataset's documentation
        class_names = [
            'T-shirt/top',
            'Trouser',
            'Pullover',
            'Dress',
            'Coat',
            'Sandal',
            'Shirt',
            'Sneaker',
            'Bag',
            'Ankle boot'
        ]

        # Print the class names
        st.write(class_names)

        classifier = ''
        if selected_model == 0:    
            report = """K-Nearest Neighbors (KNN) can achieve decent accuracy on the Fashion 
            MNIST dataset for image classification of clothing items. However, it can be
              computationally expensive due to needing distance comparisons between the test 
              image and all training images. Additionally, choosing the optimal value for 
              k (number of neighbors) and potential benefits of dimensionality reduction 
              techniques for high-dimensional image data are factors to consider for 
              optimal performance."""
            classifier = 'K-Nearest Neighbor'
        elif selected_model == 1:   # Random Forest
            report = """SVMs generally achieve good accuracy on the Fashion MNIST dataset, 
            correctly classifying a significant portion of the clothing items 
            (usually above 80%). However, their performance can be influenced by factors like 
            hyperparameter tuning and may be surpassed by other models like convolutional 
            neural networks (CNNs) specifically designed for image recognition tasks."""
            classifier = 'Support Vector Machine'
        else:        
            report = """Naive Bayes achieves decent accuracy on the Fashion MNIST dataset, 
            likely around 70-80%, due to its simplicity and efficiency. However, more complex 
            models like convolutional neural networks can achieve significantly higher 
            accuracy due to their ability to capture the intricate patterns in the 
            clothing images."""
            classifier = "Naive Bayes"

        st.subheader('Performance of the ' + classifier)

        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        y_train= st.session_state['y_train']
        y_test = st.session_state['y_test']

        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)

        st.subheader('Confusion Matrix')
        st.write('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        st.text(cm)

        st.subheader('Performance Metrics')
        st.text(classification_report(y_test, y_test_pred))
        with st.expander("Click to view classifier description"):
            st.write(classifier)
            st.write(report)


#run the app
if __name__ == "__main__":
    app()
