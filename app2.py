# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 21:22:07 2025

@author: Dell
"""
import streamlit as st
import numpy as np
import pickle 


#loading the saved model
#vectorizer=pickle.load(open('C:/Users/Dell/Desktop/P1/Vectorizer.sav','rb'))
loaded_model = pickle.load(open('C:/Users/Dell/Desktop/P1/trained_model.sav','rb'))
vectorizer = pickle.load(open('Vectorizer.sav', 'rb'))


st.title("Real or Fake news Analysis")
html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit Real and Fake News Analysis app </h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)
st.write("Enter a news Article below")

news_input = st.text_area("News Article:","")

if st.button("Predict"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = loaded_model.predict(transform_input)
        
        
        if prediction[0]==1:
            st.success("The News is Real ")
        else:
            st.error("The News is Fake ")
    else:
        st.warning("Please enter some text to analize. ")
        
        
if st.button("About"):
    st.text("This app will help us to get Real or Fake news")