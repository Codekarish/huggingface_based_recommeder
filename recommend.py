import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from bs4 import BeautifulSoup
import tensorflow as tf
import requests
import warnings
from transformers import pipeline,AutoTokenizer, AutoModelForSeq2SeqLM
import re
import json
from waitress import serve
from geopy.distance import geodesic
import os
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Your OpenCage API key
OPENCAGE_API_KEY = os.getenv('OPENCAGE_API_KEY')

# Initialize a text generation pipeline with a Hugging Face model
model_name = "gpt2"  # You can replace this with another model like 'gpt-neo', 'gpt-j', etc.
nlp_pipeline = pipeline("text-generation", model=model_name)
def fetch_data(url, params, min_results=100):
    all_properties = []
    current_page = 1
    while len(all_properties) < min_results:
        try:
            params['page'] = current_page
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            properties = data.get('data', [])
            if not properties:
                break  # Exit loop if no more data
            all_properties.extend(properties)
            current_page += 1
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break
    return pd.DataFrame(all_properties)

def extract_numeric_bedrooms(description):
    if pd.isnull(description):
        return np.nan
    if 'studio' in description.lower() or 'bedsitter' in description.lower():
        return 0  # Assuming studio/bedsitter has 0 bedrooms
    match = re.search(r'\d+', description)
    if match:
        return int(match.group(0))
    elif '+' in description:
        return 5  # For '5+ bedroom', assume max value
    return np.nan

def clean_data(df):
    if 'description' in df.columns:
        df['description'] = df['description'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text() if pd.notnull(x) else "")
    if 'bedrooms' in df.columns:
        df['bedrooms'] = df['bedrooms'].apply(extract_numeric_bedrooms)
    df = df.fillna('')
    return df

def convert_bedrooms_to_integers(df):
    def parse_bedroom_value(value):
        if pd.isnull(value):
            return None
        if isinstance(value, str):
            if '+' in value:
                return 5
            match = re.search(r'\d+', value)
            if match:
                return int(match.group())
            else:
                return None
        if isinstance(value, (float, int)):
            return int(value)
        return None

    df['bedrooms'] = df['bedrooms'].apply(parse_bedroom_value)
    df['bedrooms'] = df['bedrooms'].astype('Int64')
    return df

def preprocess_text(df, query_keywords):
    def weight_keywords(text, keywords):
        for keyword in keywords:
            text = re.sub(f"(?i){keyword}", f"{keyword} {keyword}", text)
        return text

    df['weighted_text'] = df['name'] + ' ' + df['description']
    df['weighted_text'] = df['weighted_text'].apply(lambda x: weight_keywords(x, query_keywords))

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['weighted_text'])
    return vectorizer, X

def encode_categorical_data(df):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    categorical_data = df[['property_type', 'listing_type', 'location']]
    encoded_categorical_data = encoder.fit_transform(categorical_data)
    return encoder, encoded_categorical_data

def combine_features(tfidf_matrix, encoded_categorical_data):
    return np.hstack((tfidf_matrix.toarray(), encoded_categorical_data))

def extract_keywords(query):
    keywords = re.findall(r'\b\w+\b', query.lower())
    return keywords

def generate_prompt(query):
    return f"""
    You are a helpful assistant for a property recommendation system. Your task is to understand the user query and map it to the correct attributes from a list of property descriptions. Given the following user query, please extract the key attributes including price, number of bedrooms, location, and agent name if mentioned. Provide the information in a structured format.

    User Query: "{query}"

    Attributes:
    """

def interpret_query_with_huggingface(query):
    prompt = generate_prompt(query)
    response = nlp_pipeline(prompt, max_length=150)
    return json.loads(response[0]['generated_text'])

def get_lat_lon(location):
    url = f'https://api.opencagedata.com/geocode/v1/json?q={location}&key={OPENCAGE_API_KEY}'
    response = requests.get(url)
    data = response.json()
    if data['results']:
        return data['results'][0]['geometry']['lat'], data['results'][0]['geometry']['lng']
    return None, None

def get_distance(loc1, loc2):
    return geodesic(loc1, loc2).km

def get_recommendations(query, vectorizer, tfidf_matrix, encoder, encoded_categorical_data, df, k=100):
    price, bedrooms, location, agent_name = parse_query(query)
    query_keywords = extract_keywords(query)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    if price:
        df['price_amount'] = df['price'].apply(lambda x: x['price'] if isinstance(x, dict) and 'price' in x else 0)
        df = df[(df['price_amount'] <= price * 1.1) & (df['price_amount'] >= price * 0.9)]

    if location:
        user_lat, user_lon = get_lat_lon(location)
        if user_lat is not None and user_lon is not None:
            user_location = (user_lat, user_lon)
            df['distance'] = df.apply(lambda row: get_distance(user_location, (row['lat'], row['lon'])), axis=1)
            df = df.sort_values(by='distance')

    if bedrooms:
        if 'bedrooms' in df.columns:
            df['bedroom_diff'] = df['bedrooms'].apply(lambda x: abs(x - bedrooms) if pd.notnull(x) else np.nan)
            df = df.sort_values(by='bedroom_diff')

    if agent_name:
        if 'agent' in df.columns:
            df['agent_match'] = df['agent'].apply(lambda x: x['name'].strip().lower() if isinstance(x, dict) and 'name' in x else 'unknown')
            df = df[df['agent_match'] == agent_name]

    weights = {
        'location': 5,
        'property_type': 4,
        'size': 3,
        'submission_type': 2,
        'price': 1,
        'creation_date': 0.5,
        'property_setting': 0.5,
        'customer_modifiers': 0.5,
        'user_metrics': 0.5,
    }

    weighted_similarities = similarities.copy()

    for i, row in df.iterrows():
        if 'location' in query_keywords and 'location' in df.columns:
            weighted_similarities[i] += weights['location'] * 0.1
        if 'property_type' in query_keywords and 'property_type' in df.columns:
            weighted_similarities[i] += weights['property_type'] * 0.1
        if 'size' in query_keywords and 'size' in df.columns:
            weighted_similarities[i] += weights['size'] * 0.1
        if 'submission_type' in query_keywords and 'submission_type' in df.columns:
            weighted_similarities[i] += weights['submission_type'] * 0.1
        if 'price' in query_keywords and 'price_amount' in df.columns:
            weighted_similarities[i] += weights['price'] * 0.1
        if 'creation_date' in query_keywords and 'created_at' in df.columns:
            weighted_similarities[i] += weights['creation_date'] * 0.05
        if 'property_setting' in query_keywords and 'property_setting' in df.columns:
            weighted_similarities[i] += weights['property_setting'] * 0.05
        if 'customer_modifiers' in query_keywords and 'customer_modifiers' in df.columns:
            weighted_similarities[i] += weights['customer_modifiers'] * 0.05
        if 'user_metrics' in query_keywords and 'user_metrics' in df.columns:
            weighted_similarities[i] += weights['user_metrics'] * 0.05

    df['similarity_score'] = weighted_similarities
    top_k_indices = np.argsort(-weighted_similarities)[:k]
    top_k_df = df.iloc[top_k_indices]

    return top_k_df

def parse_query(query):
    response = interpret_query_with_huggingface(query)
    price = response.get('price')
    bedrooms = response.get('bedrooms')
    location = response.get('location')
    agent_name = response.get('agent_name')
    return price, bedrooms, location, agent_name
def main(query):
    url = "https://sapi.hauzisha.co.ke/api/properties/search"
    params = {
        "per_page": 300
    }

    df = fetch_data(url, params)
    df = clean_data(df)
    df = convert_bedrooms_to_integers(df)

    query_keywords = extract_keywords(query)
    vectorizer, tfidf_matrix = preprocess_text(df, query_keywords)
    encoder, encoded_categorical_data = encode_categorical_data(df)
    combined_features = combine_features(tfidf_matrix, encoded_categorical_data)
    
    recommendations = get_recommendations(query, vectorizer, tfidf_matrix, encoder, encoded_categorical_data, df)
    return recommendations

if __name__ == "__main__":
    sample_query = "2 Bedroom house for rent at Ruiru for 50000 KSh"
    recommendations = main(sample_query)
    print("\nRecommended Properties:")
    for rec in recommendations:
        print(f"ID: {rec['id']}, Submission Type: {rec['submission_type']}, Bedrooms: {rec['bedrooms']}, Agent: {rec['agent_name']}, Price: {rec['price_amount']} {rec['price_label']} ({rec['price_currency']}), Description: {rec['formatted_description'][:100]}..., Score: {rec['score']}")