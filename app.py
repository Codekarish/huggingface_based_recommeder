from flask import Flask, request, jsonify, render_template
import recommend

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_route():
    query = request.form.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Call the recommendation function directly from recommend.py
    recommendations = recommend.get_recommendations(query)
    
    # Convert the recommendations to a dictionary format suitable for JSON response
    recommendations = [rec.to_dict() for rec in recommendations]
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
