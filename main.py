import streamlit as st
import numpy as np
import pandas as pd
st.title("Iris Prediction App")
st.markdown("""
### This app predicts Iris flower type""")
sepal_length=st.sidebar.header("User input parameters")
def userinput():
    sepal_length=st.sidebar.slider("Sepal Length",4.2,7.7)
    sepal_width=st.sidebar.slider("Sepal Width",2.0,4.4)
    petal_length=st.sidebar.slider("Petal Length",1.0,6.9)
    petal_width=st.sidebar.slider("Petal Width",0.1,2.5)
    data={
        "Sepal Length":sepal_length,
        "Sepal Width":sepal_width,
        "Petal Length":petal_length,
        "Petal Width":petal_width
    }
    features=pd.DataFrame(data,index=[0])
    return features
df=userinput()
st.subheader("User input paramter")
st.write(df)

from sklearn import datasets
iris=datasets.load_iris()
X=iris.data
y=iris.target
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(X,y)
prediction=clf.predict(df)
prediction_prob=clf.predict_proba(df)
st.subheader("Class labels and their corresponding index")
st.write(iris.target_names)

st.subheader("Prediction")
st.write(iris.target_names[prediction])

st.subheader("Prediction probability")
st.write(prediction_prob)
