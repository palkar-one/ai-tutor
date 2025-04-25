import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import json
from typing import List, Dict
from langchain.chat_models import init_chat_model


def login_required():
    if not st.session_state.get("logged_in", False):
        st.warning("You must log in to access this page.")
        st.stop()

# from app import authenticated_page

if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'questions_generated' not in st.session_state:
    st.session_state.questions_generated = False
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'score' not in st.session_state:
    st.session_state.score = None

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "API-KEY"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

def extract_text_from_pdfs(pdf_files):
    """Extract text from uploaded PDF files"""
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def split_text_into_chunks(text):
    """Split text into chunks using recursive character text splitter"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

def create_vector_store(chunks):
    """Create FAISS vector store from document chunks"""
    return FAISS.from_documents(chunks, embeddings)

def generate_questions(topic: str, vector_store, num_questions=10):
    """Generate multiple choice questions from the PDF content"""
    # Retrieve relevant chunks
    docs = vector_store.similarity_search(topic, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Prompt for question generation
    prompt = PromptTemplate.from_template(
        """You are an expert quiz maker. Generate {num_questions} multiple choice questions based on the following context.
        Each question should have 4 options (a, b, c, d) with one correct answer.
        Return ONLY the raw JSON output without any additional text or formatting, the output must be the following structure:
        {{
            "questions": [
                {{
                    "question": "question text",
                    "options": {{
                        "a": "option a",
                        "b": "option b",
                        "c": "option c",
                        "d": "option d"
                    }},
                    "correct_answer": "a/b/c/d"
                }}
            ]
        }}

        Context: {context}
        Topic: {topic}
        """
    )
    
    chain = prompt | llm
    response = chain.invoke({
        "num_questions": num_questions,
        "context": context,
        "topic": topic
    })
    print("\n\n Response: ", response.content)
    
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        st.error("Failed to parse questions. Please try again.")
        return None

def display_quiz(questions_data):
    """Display the quiz and collect user answers"""
    st.session_state.user_answers = {}
    for i, question in enumerate(questions_data["questions"], 1):
        st.write(f"**Question {i}:** {question['question']}")
        options = question['options']
        
        # Store the correct answer for evaluation
        st.session_state.user_answers[f"q{i}_correct"] = question['correct_answer']
        
        # Display options and collect user answer
        user_answer = st.radio(
            f"Select your answer for Question {i}:",
            options=["a", "b", "c", "d"],
            key=f"q{i}",
            format_func=lambda x: f"{x}) {options[x]}"
        )
        st.session_state.user_answers[f"q{i}_user"] = user_answer
        st.write("---")

def evaluate_answers():
    """Evaluate user answers and calculate score"""
    correct = 0
    total = len([k for k in st.session_state.user_answers.keys() if k.startswith("q") and k.endswith("_correct")])
    
    for i in range(1, total + 1):
        user_answer = st.session_state.user_answers.get(f"q{i}_user")
        correct_answer = st.session_state.user_answers.get(f"q{i}_correct")
        if user_answer == correct_answer:
            correct += 1
    
    st.session_state.score = (correct / total) * 100 if total > 0 else 0
    return st.session_state.score

# @authenticated_page
def main():
    st.title("📝 PDF Quiz Generator")
    st.write("Upload PDFs, ask about a topic, and get a quiz!")

    # Step 1: PDF Upload and Processing
    with st.expander("Step 1: Upload PDF Files", expanded=True):
        pdf_files = st.file_uploader(
            "Upload PDF files", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        if pdf_files and st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                text = extract_text_from_pdfs(pdf_files)
                chunks = split_text_into_chunks(text)
                st.session_state.vector_store = create_vector_store(chunks)
                st.session_state.pdf_processed = True
                st.success("PDFs processed successfully!")

    # Step 2: Question Generation
    if st.session_state.pdf_processed:
        with st.expander("Step 2: Generate Questions", expanded=True):
            topic = st.text_input(
                "Enter a topic from the PDF to generate questions about:",
                placeholder="e.g., machine learning, history of computing"
            )
            
            if topic and st.button("Generate Quiz"):
                with st.spinner("Generating questions..."):
                    questions_data = generate_questions(topic, st.session_state.vector_store)
                    if questions_data:
                        st.session_state.questions_data = questions_data
                        st.session_state.questions_generated = True
                        st.success("Quiz generated successfully!")

    # Step 3: Take the Quiz
    if st.session_state.questions_generated:
        with st.expander("Step 3: Take the Quiz", expanded=True):
            st.write("### Test Your Knowledge")
            display_quiz(st.session_state.questions_data)
            
            if st.button("Submit Answers"):
                score = evaluate_answers()
                st.success(f"Your score: {score:.1f}%")
                
                # Show correct answers
                st.write("### Correct Answers:")
                for i, question in enumerate(st.session_state.questions_data["questions"], 1):
                    correct_option = st.session_state.user_answers[f"q{i}_correct"]
                    st.write(f"Question {i}: Correct answer is {correct_option}) {question['options'][correct_option]}")

    # Reset button
    if st.session_state.pdf_processed and st.button("Start Over"):
        st.session_state.clear()
        st.rerun()

if __name__ == "__main__":
    login_required()
    main()