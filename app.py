from flask import Flask, request, jsonify, render_template
import recommend

app = Flask(__name__)

# Initialize your model and variables
vectorizer, tfidf_matrix = recommend.preprocess_text()  # Adjust according to your setup
encoder, encoded_categorical_data = recommend.encode_categorical_data()
df = recommend.fetch_data()  # Make sure to provide the URL and params if necessary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_route():
    query = request.form.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    recommendations = recommend.get_recommendations(query, vectorizer, tfidf_matrix, encoder, encoded_categorical_data, df)
    recommendations = [rec.to_dict() for rec in recommendations]
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
