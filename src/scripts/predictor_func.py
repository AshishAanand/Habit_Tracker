import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer, util

# A single function that tells you weither your actions match your goals or not

def predict(user_goal: str, habit_log: str, threshold: float) -> str:
    similarity = ""
    sentiment = ""

    # 1. Convert inputs into pandas Dataframe
    data = {
        'user_goal': [user_goal],
        'habit_log': [habit_log]
    }
    df = pd.DataFrame(data)
    # 2. load ML models
    habit_type_model = joblib.load('../models/habit_type_predictor.pkl')
    sentiment_model = joblib.load('../models/sentiment_predictor.pkl')
    # 3. pass the dataframe to first model to predict habit_type
    df['predicted_habit_type'] = habit_type_model.predict(df[['user_goal', 'habit_log']])
    # 4. Then add the predicted value in the dataframe
    df['predicted_habit_type'] = df['predicted_habit_type'].astype(str)
    # 5. Last pass the new dataframe to the last model to predict sentiment
    df['predicted_sentiment'] = sentiment_model.predict(df[['user_goal', 'habit_log', 'predicted_habit_type']])
    # 6. Return the predicted sentiment
    predicted_sentiment = df['predicted_sentiment'].values[0]
    if predicted_sentiment == 'positive':
        sentiment = "You are doing well! Keep it up!"
    elif predicted_sentiment == 'negative':
        sentiment = "You need to improve your actions to match your goals."
    # 6. Find similarity between goals and actions and tell user weither user is doing well or not.
    user_goal_embedding = SentenceTransformer('all-MiniLM-L6-v2').encode(user_goal, convert_to_tensor=True)
    habit_log_embedding = SentenceTransformer('all-MiniLM-L6-v2').encode(habit_log, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(user_goal_embedding, habit_log_embedding)
    if cosine_score >= threshold:  # Threshold for similarity
        similarity = "Your actions are aligned with your goals."
    else:
        similarity = "Your actions are not aligned with your goals. Please review your habit log."
    
    return sentiment, similarity