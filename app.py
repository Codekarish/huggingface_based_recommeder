from flask import Flask, request, jsonify, render_template
import pandas as pd
from recommend import fetch_data, clean_data, convert_bedrooms_to_integers, preprocess_text, encode_categorical_data, combine_features, get_recommendations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Configure API URL and parameters
API_URL = 'https://api.example.com/properties'
API_PARAMS = {'key': 'your_api_key', 'type': 'rental'}

# Fetch and prepare data
df = fetch_data(API_URL, API_PARAMS)
df = clean_data(df)
df = convert_bedrooms_to_integers(df)
vectorizer, tfidf_matrix = preprocess_text(df, ['property', 'apartment', 'bedroom'])  # Example keywords
encoder, encoded_categorical_data = encode_categorical_data(df)
combined_features = combine_features(tfidf_matrix, encoded_categorical_data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.form.get('query')
    recommendations = get_recommendations(query, vectorizer, tfidf_matrix, encoder, encoded_categorical_data, df)
    recommendations = recommendations.to_dict(orient='records')
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
