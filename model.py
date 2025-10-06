import pickle
import pandas as pd
import numpy as np

# Load pickled items once
sentiment_model = pickle.load(open('sentiment_model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
user_final_rating = pickle.load(open('user_final_rating.pkl', 'rb'))
df_clean = pickle.load(open('df_clean.pkl', 'rb'))

# Function to get top 5 recommended products for user

def product_recommendations_user(user_name):
    if user_name not in user_final_rating.index:
        return None
    top20_products = list(user_final_rating.loc[user_name].sort_values(ascending=False).head(20).index)
    df_top20 = df_clean[df_clean['name'].isin(top20_products)].drop_duplicates(subset=['cleaned_review']).copy()
    # TF-IDF transform
    X_tfidf = tfidf_vectorizer.transform(df_top20['cleaned_review'])
    preds = sentiment_model.predict(X_tfidf)
    df_top20['predicted_sentiment'] = preds
    df_top20['positive_sentiment'] = df_top20['predicted_sentiment'].apply(lambda x: 1 if x == 1 else 0)
    pred_df = df_top20.groupby('name').agg(pos_sent_count=('positive_sentiment', 'sum'), total_sent_count=('predicted_sentiment', 'count')).reset_index()
    pred_df['pos_sent_percentage'] = (pred_df['pos_sent_count'] / pred_df['total_sent_count']) * 100
    top5 = pred_df.sort_values(by='pos_sent_percentage', ascending=False).head(5)[['name', 'pos_sent_percentage']]
    return top5
