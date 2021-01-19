import streamlit as st
from sklearn import datasets
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import     SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score
from sklearn.decomposition   import PCA
import matplotlib.pyplot as plt

### Functions ###
def get_dataset(Selector):
    """ Loads Dataset Accordint to User input  (Managed With SlideBars) """
    if Selector == "Iris":
        data = datasets.load_iris()
        

    elif Selector == "Breast Cancer":
        data = datasets.load_breast_cancer()

    elif Selector == "Wine Dataset":
        data = datasets.load_wine()
    
    X = data.data 
    y = data.target
  

    st.write(f"Dataset  Loaded with")
    st.write("Shape of X ",X.shape[0],"examples and ",X.shape[1],"features")
    st.write("Shape of y ",y.shape[0]," examples ")
    
    
    return X,y




def get_classifier(Selector):
    """ Select Paramaters needed for Classifier According to User input  (Managed With SlideBars)  """
    params = dict()
    if Selector == "KNN":
        K = st.sidebar.slider("K",1,10)
        params["K"] = K
        
    elif Selector == "SVM":
        C = st.sidebar.slider("C",0.01,10.0)
        params["C"] = C
        
    elif Selector == "Random Forest":
        max_depth = st.sidebar.slider("Max Depth",2,15)
        n_estimators = st.sidebar.slider("N estimators",1,10)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators

    return params

def init_classifier(Selector,params):
    if Selector == "KNN":
        classifier = KNeighborsClassifier(n_neighbors=params["K"]) 
        
    elif Selector == "SVM":
        classifier = SVC(C=params["C"])
        
    elif Selector == "Random Forest":
        classifier = RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"])
        
    return classifier

def plot(x,y,p,title1,title2):
    pca = PCA(n_components=2)           # reduce to 2 dimensions
    X_projeceted = pca.fit_transform(x) # Apply Principal Component Analysis

    x1 = X_projeceted[:,0]  # Get the First Component
    x2 = X_projeceted[:,1]  # Get The Second Cponent

    fig,ax= plt.subplots(nrows=1,ncols=2,figsize = (10,5))      # Create Figure 
    ax[0].scatter(x1,x2,c=y, alpha=0.8,cmap="viridis")     # Create Scatter Plot
    ax[0].set_xlabel("Principal Component 1")              # Name X axis
    ax[0].set_ylabel("Principal Component 2")              # Name y axis
    ax[0].set_title(title1)                                 # Set title
   

    ax[1].scatter(x1,x2,c=p, alpha=0.8,cmap="viridis")     # Create Scatter Plot
    ax[1].set_xlabel("Principal Component 1")              # Name X axis
    ax[1].set_ylabel("Principal Component 2")              # Name y axis
    ax[1].set_title(title2)                                 # Set title
    plt.tight_layout()
    



    st.pyplot(fig)                                   # This is used instead of plt.show() so that we can show our plot in the App

def run_training(classifier,X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    acc = accuracy_score(y_test,y_predict)
    st.write("Accuracy",acc)
   
    plot(X_test,y_test,y_predict,"Original",f"Predictions with accuracy {round(acc,2)}")
    return 0 
