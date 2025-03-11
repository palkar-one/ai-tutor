from flask import Flask, request, jsonify
from flask_cors import CORS
import bcrypt
import sqlite3
import os
import fitz
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import requests

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)

# Database setup (SQLite)
def create_table():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

create_table()

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not name or not email or not password:
        return jsonify({'message': 'Missing required fields'}), 400

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)', (name, email, hashed_password))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Signup successful'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'message': 'Email already exists'}), 409
    except Exception as e:
        return jsonify({'message': 'An error occurred: ' + str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'message': 'Missing required fields'}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            return jsonify({'message': 'Login successful'}), 200
        else:
            return jsonify({'message': 'Invalid credentials'}), 401
    except Exception as e:
        return jsonify({'message': 'An error occurred: ' + str(e)}), 500

@app.route('/test')
def test():
    return jsonify({'message': 'Test route working'})

@app.route('/')
def index():
    return app.send_static_file(os.path.join(app.root_path, 'static', 'index.html'))

# Summarization endpoint
@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        topic = request.form.get('topic')
        file = request.files.get('file')

        if not topic or not file:
            return jsonify({'message': 'Missing required fields'}), 400

        text = extract_text(file)
        structured_text = split_by_headings(text)
        summary = generate_summary(structured_text, topic)

        return jsonify({'summary': summary}), 200
    except Exception as e:
        return jsonify({'message': 'An error occurred: ' + str(e)}), 500

# Question Answering Endpoint
@app.route('/answer_question', methods=['POST'])
def answer_question():
    try:
        question = request.form.get('question')
        file = request.files.get('file')

        if not question or not file:
            return jsonify({'message': 'Missing required fields'}), 400

        text = extract_text(file)
        structured_text = split_by_headings(text)
        answer = generate_answer(structured_text, question)

        return jsonify({'answer': answer}), 200
    except Exception as e:
        return jsonify({'message': 'An error occurred: ' + str(e)}), 500

# Function to extract text from the uploaded file
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

# Function to detect headings and structure the text
def split_by_headings(text):
    sections = {}
    current_heading = "General"
    sections[current_heading] = ""

    for line in text.split("\n"):
        if re.match(r"^(\d+\.\d+|[A-Z][A-Za-z0-9\s-]+|[A-Z]+[:])$", line.strip()):
            current_heading = line.strip()
            sections[current_heading] = ""
        elif current_heading:
            sections[current_heading] += line + "\n"

    return sections

# Function to generate summary using RAG, FAISS, and Mistral AI (via OpenRouter)
def generate_summary(structured_text, topic):
    OPENROUTER_API_KEY = "YOUR_API_KEY"

    model_embedding = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    d = 384
    index = faiss.IndexFlatL2(d)
    section_embeddings = {}
    section_list = list(structured_text.keys())

    embeddings_list = []
    for heading, content in structured_text.items():
        embedding = model_embedding.encode(content)
        section_embeddings[heading] = embedding
        embeddings_list.append(embedding)

    if embeddings_list:
        index.add(np.array(embeddings_list))

    def search_topic(query, include_subtopics=True, top_k=5):
        query_embedding = model_embedding.encode([query])
        distances, indices = index.search(query_embedding, top_k)
        retrieved_texts = []

        for i, distance in zip(indices[0], distances[0]):
            if i < len(section_list) and distance < 1.2:
                heading = section_list[i]
                retrieved_text = structured_text[heading]
                if i + 1 < len(section_list):
                    retrieved_text += "\n" + structured_text[section_list[i + 1]]
                if i - 1 >= 0:
                    retrieved_text = structured_text[section_list[i - 1]] + "\n" + retrieved_text
                retrieved_texts.append(f"### {heading}\n" + retrieved_text)

        if not retrieved_texts:
            return "No relevant information found in the book."
        return "\n\n".join(retrieved_texts)

    relevant_text = search_topic(topic)
    if relevant_text == "No relevant information found in the book.":
        return relevant_text

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = (
        "Summarize the following text efficiently for students. Extract key concepts, explain them clearly, "
        "remove unnecessary information, and format the summary for easy studying. Ensure coherence and completeness.\n\n"
        f"Text: {relevant_text}\n\nStructured Summary:"
    )
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)

    if response.status_code != 200:
        print(f"API Error: {response.status_code}, {response.text}")
        return f"Error: {response.status_code}, {response.text}"

    print(response.json())

    summary = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No summary generated.")
    return summary

# Function to generate answer using RAG, FAISS, and Mistral AI (via OpenRouter)
def generate_answer(structured_text, question):
    OPENROUTER_API_KEY = "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"

    model_embedding = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    d = 384
    index = faiss.IndexFlatL2(d)
    section_embeddings = {}
    section_list = list(structured_text.keys())

    embeddings_list = []
    for heading, content in structured_text.items():
        embedding = model_embedding.encode(content)
        section_embeddings[heading] = embedding
        embeddings_list.append(embedding)

    if embeddings_list:
        index.add(np.array(embeddings_list))

    def search_topic(query, include_subtopics=True, top_k=5):
        query_embedding = model_embedding.encode([query])
        distances, indices = index.search(query_embedding, top_k)
        retrieved_texts = []

        for i, distance in zip(indices[0], distances[0]):
            if i < len(section_list) and distance < 1.2:
                heading = section_list[i]
                retrieved_text = structured_text[heading]
                if i + 1 < len(section_list):
                    retrieved_text += "\n" + structured_text[section_list[i + 1]]
                if i - 1 >= 0:
                    retrieved_text = structured_text[section_list[i - 1]] + "\n" + retrieved_text
                retrieved_texts.append(f"### {heading}\n" + retrieved_text)

        if not retrieved_texts:
            return "No relevant information found in the book."
        return "\n\n".join(retrieved_texts)

    relevant_text = search_topic(question)
    if relevant_text == "No relevant information found in the book.":
        return relevant_text

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = (
        "Answer the following question based on the provided text. \n\n"
        f"Text: {relevant_text}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    payload = {
        "model": "mistralai/Mistral-Small-24B-Instruct-2501", # Model changed to Mistral-Small
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)

    if response.status_code != 200:
        print(f"API Error: {response.status_code}, {response.text}")
        return f"Error: {response.status_code}, {response.text}"

    print(response.json())

    answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No answer generated.")
    return answer

if __name__ == '__main__':
    app.run(debug=True)