from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)

OPENROUTER_API_KEY = "YOUR_API_KEY" #Replace with your actual API key

def extract_text(file):
    text = ""
    if file.filename.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text("text") + "\n"
    elif file.filename.endswith(".txt"):
        text = file.read().decode('utf-8')
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or TXT file.")
    return text

def create_faiss_index(text):
    model_embedding = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sentences = text.split('\n')
    embeddings = model_embedding.encode(sentences)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, sentences, model_embedding

def search_faiss(index, sentences, model_embedding, query, top_k=5):
    query_embedding = model_embedding.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    retrieved_texts = [sentences[i] for i in indices[0]]
    return retrieved_texts

def generate_answer(relevant_texts, question):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = (
        "Answer the following question based on the provided text. \n\n"
        f"Text: {' '.join(relevant_texts)}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
    if response.status_code != 200:
        return f"Error: {response.status_code}, {response.text}"
    answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No answer generated.")
    return answer

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/answer_question', methods=['POST'])
def answer_question():
    try:
        question = request.form.get('question')
        file = request.files.get('file')

        if not question or not file:
            return jsonify({'message': 'Missing required fields'}), 400

        text = extract_text(file)
        index, sentences, model_embedding = create_faiss_index(text)
        relevant_texts = search_faiss(index, sentences, model_embedding, question)
        answer = generate_answer(relevant_texts, question)
        return jsonify({'answer': answer}), 200
    except Exception as e:
        return jsonify({'message': 'An error occurred: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)