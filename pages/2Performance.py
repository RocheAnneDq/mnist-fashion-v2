#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():

    st.sidebar.subheader('Select the classifier')
    # Create the selecton of classifier
    clf = KNeighborsClassifier(n_neighbors=5)
    options = ['K Nearest Neighbor', 'Support Vector Machine', 'Naive Bayes']
    selected_option = st.sidebar.selectbox('Select the classifier', options)
    if selected_option =='Support Vector Machine':
        clf = SVC(kernel='linear')
        st.session_state['selected_model'] = 1
    elif selected_option=='Naive Bayes':        
        clf = GaussianNB()
        st.session_state['selected_model'] = 2
    else:
        clf = KNeighborsClassifier(n_neighbors=5)
        st.session_state['selected_model'] = 0


    classifier = ''
    if st.session_state['selected_model'] == 0:     # decision tree
        report = """Achieves good accuracy, but can be prone to 
        overfitting, leading to lower performance on unseen data.
        Simple and interpretable, allowing visualization of decision rules.
        Susceptible to changes in the training data, potentially 
        leading to high variance in predictions."""
        classifier = 'Decision Tree'
    elif st.session_state['selected_model'] == 1:   # Random Forest
        report = """Generally outperforms a single decision tree, 
        reaching accuracy close to 98%. Reduces overfitting through 
        averaging predictions from multiple trees. Ensemble method - 
        combines predictions from multiple decision trees, leading to 
        improved generalization and reduced variance. Less interpretable 
        compared to a single decision tree due to the complex 
        ensemble structure."""
        classifier = 'Random Forest'
    elif st.session_state['selected_model'] == 2:   # Extreme Random Forest
        report = """Performance: Can achieve similar or slightly better 
        accuracy compared to a random forest, but results can vary 
        depending on hyperparameter tuning. Introduces additional randomness 
        during tree building by randomly selecting features at each split.  Aims to 
        further improve generalization and reduce overfitting by increasing 
        the diversity of trees in the ensemble. Requires careful 
        hyperparameter tuning to achieve optimal performance."""
        classifier = "Extreme Random Forest"
    else:
        report = """Accuracy: While KNN can achieve reasonable accuracy (around 80-90%), 
        it's often outperformed by more sophisticated models like Support 
        Vector Machines (SVMs) or Convolutional Neural Networks (CNNs) which can 
        reach over 97% accuracy.\nComputational cost: Classifying new data points 
        requires comparing them to all data points in the training set, making 
        it computationally expensive for large datasets like MNIST."""
        classifier = "K-Nearest Neighbor"

    st.subheader('Performance of the ' + classifier)

    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train= st.session_state['y_train']
    y_test = st.session_state['y_test']

    clf = st.session_state['clf']
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)

    st.subheader('Confusion Matrix')
    st.write('Confusion Matrix')
    cm = confusion_matrix(y_test, y_test_pred)
    st.text(cm)

    st.subheader('Performance Metrics')
    st.text(classification_report(y_test, y_test_pred))
    
    st.write(classifier)
    st.write(report)

    # save the clf to the session state
    st.session_state['clf'] = clf

#run the app
if __name__ == "__main__":
    app()
