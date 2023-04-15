from transformers import AutoTokenizer, CamembertForSequenceClassification
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = CamembertForSequenceClassification.from_pretrained('./models/model_1')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # predicted_label = torch.argmax(outputs.logits).item()
    probabilities = torch.softmax(logits, dim=1)

    # Convert probabilities to percentages
    percentages = probabilities * 100
    return jsonify({'percentages': percentages.tolist()})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response





if __name__ == '__main__':
    app.run(debug=True)