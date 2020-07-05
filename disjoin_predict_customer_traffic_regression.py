#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing the packages

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Disjoin: An app to predict the foot traffic in stores")

#import dataset
st.markdown("### Browse dataset")
data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
dataset = pd.read_csv(data)

st.sidebar.title("Explore")
sb = st.sidebar.selectbox("Choose the options",("About","Top 5 records","Summary"))


if sb == "About":
    st.markdown("This is a machine application built using streamlit. It has following features")
    st.markdown("1. Browse the dataset")
    st.markdown("2. Select the options for performing EDA")
    st.markdown("3. Input the data from user")
    st.markdown("4. Predict the value")
    st.subheader("Explanation about the dataset")
    st.markdown("The Disjoin dataset is sampled to predict the count of customers based on historical data")

if sb == "Top 5 records":
    #View dataset
    st.markdown("### Top 5 rows of dataset")
    st.dataframe(dataset.head())

if sb == "Summary":
    #Summary of Dataset
    st.markdown("### Summary of dataset")
    st.write(dataset.describe())
    
st.sidebar.title("Visuals of Traffic")
sb1 = st.sidebar.radio("Choice",("Daily","Hourly"))

if sb1 == "Daily":
    st.markdown("### Graph: Count of Traffic vs Day")
    day = dataset['Day']
    count = dataset['Count']
    plt.bar(day,count, color = 'g')
    plt.title("Day vs Traffic")
    plt.xlabel("Day")
    plt.ylabel("Count")
    st.pyplot()
    
if sb1 == "Hourly":
    st.markdown("### Graph: Count of Traffic vs Hour")
    hour = dataset['Time']
    count = dataset['Count']
    plt.bar(hour,count, color = 'm')
    plt.title("Hour vs Traffic")
    plt.xlabel("Hour")
    plt.ylabel("Count")
    st.pyplot()

st.subheader("Prediction")
st.markdown("#### Input the values to predict")

#sb3 = st.sidebar.checkbox("Input")
st.markdown("#### Prediction based on Day and hour")
day_input = st.text_input("Enter the day")
hour_input = st.text_input("Enter the hour")
st.write("Entered day is " + day_input + " and hour is " + hour_input)


# X and Y values
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

Y = Y.reshape(len(Y),1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

# Splitting dataset into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train, Y_test = train_test_split(X,Y,test_size= 0.2, random_state=0)

#Training using SVM model
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)

#Prediction
y_count = sc_Y.inverse_transform(regressor.predict(sc_X.transform([[int(day_input),int(hour_input)]])))

st.markdown("### Predicted count is") 
st.markdown(int(y_count))

#Button to click for predict
#predict = st.button("Predict")
#st.write("#### Predicted count is" + y_count)
#st.success("Display count" + int(y_count))



    
    
    





