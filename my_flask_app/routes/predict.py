# predict.py

from flask import Blueprint, request, jsonify
from model.model import LegalRetrievalModel
from model.dataset import load_clerc_dataset
from index.faiss_index import create_index, retrieve_cases

predict_blueprint = Blueprint('predict', __name__)

model = None
index = None
cases_cleaned = None

from transformers import AutoTokenizer

@predict_blueprint.before_app_request
def load_model():
    global model, index, cases_cleaned
    # Load dataset
    dataset = load_clerc_dataset(limit=1250)
    cases = dataset['query'] + dataset['positive_passages'] + dataset['negative_passages']
    cases_cleaned = [case for case in cases if case.strip() != ""]
    
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    
    # Initialize the model with the tokenizer
    model = LegalRetrievalModel(tokenizer=tokenizer, num_cls_tokens=3)
    
    # Create the FAISS index
    index = create_index(model, cases_cleaned)
    print("Model and FAISS index loaded.")


@predict_blueprint.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Received data: {data}")  # Debugging line
        
        form_data = data['form_data']
        relevant_cases = retrieve_cases(model, index, cases_cleaned, form_data)
        
        print(f"Relevant cases: {relevant_cases}")  # Debugging line
        return jsonify({"status": "success", "relevant_cases": relevant_cases})
    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging line
        return jsonify({"status": "error", "message": str(e)})
