from utils import get_dataset, get_classifier, init_classifier,run_training
import streamlit as st

### WorkFlow ###
## https://www.youtube.com/watch?v=Klqn--Mu2pE&ab_channel=PythonEngineer 
## User ###
# Select Dataset
# Select Classifier
# Set Parameters
## Backend ##
# Split Dataset into Training and testing
# Get accuracy
# Make plots 

## Upload to Flask

## Selection Section ## 
st.title('Machine Learning App ') # Title
st.write(
        """
        ## Explote different classifiers
        which one is the best?
        """)

DataSetSelector  = st.sidebar.selectbox("Select Dataset",("Iris","Breast Cancer","Wine Dataset")) # Select Dataset
ClassifierSelector = st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random Forest"))      # Select Classifier


## Display Selection ##
st.write(f" **Dataset Selected:** {DataSetSelector}")      # Display current Selection of Dataset
X,y = get_dataset(DataSetSelector)                         # Load Dataset
st.write(f"**Classifier Selected:** {ClassifierSelector}") # Display Current classifier Selected
params = get_classifier(ClassifierSelector)                # Obtain Parameters according to Classifier (this is fed by user)
classifier = init_classifier(ClassifierSelector,params)    # Initialize Classifier with parameters
run_training(classifier,X,y)                               # Run Training and show plot 