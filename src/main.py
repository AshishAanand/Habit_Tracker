import streamlit as st
from predictor_func import predict
from streamlit.components.v1 import html
import os

if __name__ == "__main__":

    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(file_path, "r") as f:
        html_content = f.read()
    
    html(html_content,height=200, scrolling=True)

    user_goal = st.text_input("Enter your goal:")
    habit_log = st.text_area("Enter your habit log:")
    
    if st.button("Predict"):
        if user_goal and habit_log:
            sentiment, similarity = predict(user_goal, habit_log)
            st.success(sentiment)
            st.info(similarity)
        else:
            st.error("Please enter both goal and habit log.")
