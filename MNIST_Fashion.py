#Input the relevant libraries
import streamlit as st

# Define the Streamlit app
def app():
    text = """Classification of the MNIST Fashion Dataset"""
    st.header(text)

    # Use session state to track the current form

    if "clf" not in st.session_state: 
        st.session_state["clf"] = []

    if "X_train" not in st.session_state: 
        st.session_state["X_train"] = []

    if "X_test" not in st.session_state: 
        st.session_state["X_test"] = []
    
    if "y_train" not in st.session_state: 
        st.session_state["X_train"] = []
    
    if "y_test" not in st.session_state: 
        st.session_state["y_yest"] = []

    if "selected_model" not in st.session_state: 
        st.session_state["selected_model"] = 0
    
    if "mnist" not in st.session_state: 
        st.session_state["mnist"] = []

    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('mnist-fashion.png', caption="MNISt Fashion Dataset""")

    text = """Modified National Institute of Standards and Technology (MNIST) Fashion 
    dataset is a popular choice for testing and comparing machine
    learning algorithms, particularly those suited for image classification. 
    \nRelatively small size: With 70,000 images, it's computationally efficient to train and 
    test on, making it ideal for initial experimentation and algorithm evaluation.
    \nSimple image format: 
    The images are grayscale and low-resolution (28x28 pixels), 
    simplifying preprocessing and reducing computational demands.
    \nMultiple classes: 
    It consists of 10 distinct clothing categories, allowing you to assess 
    the classifiers' ability to differentiate between various categories.
    \nBenchmarking: 
    As a widely used dataset, it facilitates comparison of your models' 
    performance with established benchmarks for these algorithms on the same dataset."""
    st.write(text)

   
    
#run the app
if __name__ == "__main__":
    app()
