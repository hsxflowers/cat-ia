from flask import Blueprint, request, jsonify
from .emotion_analysis import analyze_sentiment

api = Blueprint('api', __name__)

@api.route('/emotion', methods=['POST'])
def analyze_sentiment_route():
    data = request.get_json(force=True)
    message = data['message']
    sentiment = analyze_sentiment(message)
    
    result = {
        "message": message,
        "sentiment": sentiment
    }
    
    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    
    return response
