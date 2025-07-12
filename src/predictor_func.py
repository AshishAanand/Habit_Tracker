import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer, util
import os
from functools import lru_cache

#  Load once globally (NOT inside function!)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

@lru_cache(maxsize=32)
def predict(user_goal: str, habit_log: str, threshold: float = 0.2) -> str:

    # Prepare input
    df = pd.DataFrame({'user_goal': [user_goal], 'habit_log': [habit_log]})
    
    def get_model_path(filename):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', filename)

    # Load the models
    habit_type_model = joblib.load('e:\\Learnings_and_Projects\\GitHub-repo\\Habit_Tracker\\src\\models\\habit_type_predictor.pkl')
    sentiment_model = joblib.load ('e:\\Learnings_and_Projects\\GitHub-repo\\Habit_Tracker\\src\\models\\sentiment_predictor.pkl')

    # Predict habit type
    df['predicted_habit_type'] = habit_type_model.predict(df[['user_goal', 'habit_log']])
    df['predicted_habit_type'] = df['predicted_habit_type'].astype(str)

    # Predict sentiment
    df['predicted_sentiment'] = sentiment_model.predict(df[['user_goal', 'habit_log', 'predicted_habit_type']])
    predicted_sentiment = df['predicted_sentiment'].values[0]

    # print(predicted_sentiment)

    # Convert prediction into user-friendly message
    if predicted_sentiment == 'Positive':
        sentiment = "You are doing well! Keep it up!"
    else:
        sentiment = "You need to improve your actions to match your goals."

    # Similarity Check using preloaded model
    user_goal_embedding = embedding_model.encode(user_goal, convert_to_tensor=True)
    habit_log_embedding = embedding_model.encode(habit_log, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(user_goal_embedding, habit_log_embedding)

    # print(cosine_score[[0]])

    if cosine_score <= threshold:
        similarity = "Your actions are aligned with your goals."
    else:
        similarity = "Your actions are not aligned with your goals. Please review your habit log."

    return sentiment, similarity

# #  Test run
# sentiment_msg, similarity_msg = predict("I want to learn a new language.", "I practiced Spanish for 15 minutes today.")
# print(sentiment_msg)
# print(similarity_msg)
