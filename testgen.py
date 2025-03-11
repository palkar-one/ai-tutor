from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import fitz
import faiss
from sentence_transformers import SentenceTransformer
import requests
import sqlite3

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)

OPENROUTER_API_KEY = "YOUR_API_KEY"  # Replace with your actual key

def create_connection():
    return sqlite3.connect('test_results.db')

def create_table():
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT,
                topic TEXT,
                questions TEXT,
                user_answers TEXT,
                correct_answers TEXT,
                score REAL
            )
        ''')

create_table()

def extract_text(file):
    if file.filename.endswith(".pdf"):
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            return "\n".join(page.get_text("text") for page in doc)
    elif file.filename.endswith(".txt"):
        return file.read().decode('utf-8')
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or TXT file.")

def create_faiss_index(text):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sentences = text.split('\n')
    embeddings = model.encode(sentences)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, sentences, model

def search_faiss(index, sentences, model, query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [sentences[i] for i in indices[0]]

def generate_questions(relevant_texts, topic, question_type):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"Generate 3 {question_type} questions about '{topic}' from: {' '.join(relevant_texts)}."
    payload = {"model": "mistralai/Mistral-7B-Instruct-v0.1", "messages": [{"role": "user", "content": prompt}]}
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
    response.raise_for_status()
    full_response = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No questions generated.")
    if "ANSWER:" in full_response:
        questions = "\n".join(line.split("ANSWER:")[0].strip() for line in full_response.split("\n"))
        return questions
    else:
        return full_response

def evaluate_answers(questions, user_answers, relevant_texts):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    question_list = questions.split("\n")
    user_answer_list = user_answers.split("\n")
    score = 0
    correct_answers_list =[]

    for i in range(min(len(question_list), len(user_answer_list))):
        if "ANSWER:" in question_list[i] and user_answer_list[i].strip():
            question, correct_answer = question_list[i].split("ANSWER:")
            user_answer = user_answer_list[i].strip()
            prompt = f"Q: {question}\nCorrect: {correct_answer}\nUser: {user_answer}\nContext: {' '.join(relevant_texts)}\n\nScore (1/0) and correct answer."
            payload = {"model": "mistralai/Mistral-7B-Instruct-v0.1", "messages": [{"role": "user", "content": prompt}]}
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
            try:
                response.raise_for_status()
                evaluation = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                if "Score: 1" in evaluation:
                    score += 1
                correct_answers_list.append(evaluation.split("Correct Answer:")[1].split("\n")[0].strip() if "Correct Answer:" in evaluation else "Eval Error")
            except requests.exceptions.RequestException:
                correct_answers_list.append("Eval Error")
    final_score = (score / (len(question_list) // 2)) * 100 if (len(question_list) // 2) > 0 else 0
    return final_score, "\n".join(correct_answers_list)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generate_test', methods=['POST'])
def generate_test():
    try:
        topic = request.form.get('topic')
        question_type = request.form.get('question_type')
        file = request.files.get('file')
        if not topic or not file:
            return jsonify({'message': 'Missing fields'}), 400
        text = extract_text(file)
        index, sentences, model = create_faiss_index(text)
        questions = generate_questions(search_faiss(index, sentences, model, topic), topic, question_type)
        return jsonify({'questions': questions}), 200
    except Exception as e:
        return jsonify({'message': f'Error: {e}'}), 500

@app.route('/submit_test', methods=['POST'])
def submit_test():
    try:
        topic = request.form.get('topic')
        user_answers = request.form.get('user_answers')
        file = request.files.get('file')
        if not topic or not file:
            return jsonify({'message': 'Missing fields'}), 400
        text = extract_text(file)
        index, sentences, model = create_faiss_index(text)
        relevant_texts = search_faiss(index, sentences, model, topic)
        with create_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT questions FROM test_results WHERE file_name = ? AND topic = ? ORDER BY id DESC LIMIT 1", (file.filename, topic))
            result = cursor.fetchone()
            questions = result[0] if result else generate_questions(relevant_texts, topic, "short answers")
            score, correct_answers = evaluate_answers(questions, user_answers, relevant_texts)
            cursor.execute("INSERT INTO test_results (file_name, topic, questions, user_answers, correct_answers, score) VALUES (?, ?, ?, ?, ?, ?)", (file.filename, topic, questions, user_answers, correct_answers, score))
        return jsonify({'score': score, 'correct_answers': correct_answers}), 200
    except Exception as e:
        return jsonify({'message': f'Error: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)