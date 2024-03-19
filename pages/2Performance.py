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

    classifier = ''
    if selected_model == 0:    
        report = """K-Nearest Neighbor"""
        classifier = 'K-Nearest Neighbor'
    elif selected_model == 1:   # Random Forest
        report = """Support Vector Mach8ine"""
        classifier = 'Support Vector Machine'
    else:        
        report = """Naive Bayes"""
        classifier = "Naive Bayes"

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


#run the app
if __name__ == "__main__":
    app()
