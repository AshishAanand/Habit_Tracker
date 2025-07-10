import pandas as pd
import joblib

# A single function that tells you weither your actions match your goals or not

def predict(user_goal: str, habit_log: str) -> str:
    # 1. Convert inputs into pandas Dataframe
    # 2. load ML models
    # 3. pass the dataframe to first model to predict habit_type
    # 4. Then add the predicted value in the dataframe
    # 5. Last pass the new dataframe to the last model to predict sentiment
    # 6. Find similarity between goals and actions and tell user weither user is doing well or not.
    pass