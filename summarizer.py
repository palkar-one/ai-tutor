import streamlit as st
import fitz  # PyMuPDF for PDF processing
import requests
from bs4 import BeautifulSoup
import urllib.parse
import base64
from PIL import Image
import io

# Function to extract images of PDF pages for display
def display_pdf_images(file):
    pdf_document = fitz.open(stream=io.BytesIO(file.getvalue()), filetype="pdf")
    images = []
    texts = []
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        texts.append(page.get_text("text"))
    
    return images, texts

# Function to search YouTube videos without API key using Google Search
def search_youtube(query, max_results=5):
    search_url = f"https://www.google.com/search?q=site:youtube.com+{urllib.parse.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    video_data = []
    for link in soup.select("a[href^='https://www.youtube.com/watch']")[:max_results]:
        video_url = link["href"]
        video_id = video_url.split("v=")[-1]
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
        video_data.append((video_url, thumbnail_url))
    
    return video_data

# Function to search free courses with thumbnails
def search_free_courses(query, max_results=5):
    search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}+free+course+site:coursera.org+OR+site:udemy.com+OR+site:edx.org+OR+site:khanacademy.org"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    courses = []
    for result in soup.select("a[href]")[:max_results]:
        link = result["href"]
        course_name = result.get_text(strip=True)
        courses.append((course_name, link))
    
    return courses

# Streamlit UI Layout
st.set_page_config(layout="wide")
st.title("📚 Book Viewer & Topic-Based Recommendations")

col1, col2 = st.columns([1, 2])

# Left Side: File Upload & PDF Viewer
with col1:
    uploaded_file = st.file_uploader("Upload a Book (PDF)", type=["pdf"])
    selected_text = ""
    
    if uploaded_file is not None:
        images, texts = display_pdf_images(uploaded_file)
        page_num = st.slider("Select Page", 1, len(images), 1) - 1
        st.image(images[page_num], use_container_width=True)
        
        extracted_text = texts[page_num]
        selected_text = st.text_area("Select a portion of text", extracted_text, height=100)
    
    if selected_text:
        st.session_state["selected_text"] = selected_text
    
    if "selected_text" in st.session_state:
        selected_topic = st.session_state["selected_text"]
        st.subheader("Selected Topic: " + selected_topic)
        
        col3, col4 = st.columns(2)
        with col3:
            if st.button("Highlight"):
                st.write(f"Highlighted: {selected_topic}")
        with col4:
            if st.button("Summarize"):
                st.write(f"Summary of {selected_topic} will be displayed here.")
        with col3:
            if st.button("Generate Test"):
                st.write(f"Test questions for {selected_topic} will be generated here.")
        with col4:
            if st.button("Ask a Question"):
                st.write("Answering user questions based on the uploaded content.")

# Right Side: Recommendations
with col2:
    if "selected_text" in st.session_state:
        topic = st.session_state["selected_text"]
        st.subheader(f"📌 Recommendations for: {topic}")
        
        # Fetch YouTube videos
        videos = search_youtube(topic)
        st.subheader("🎥 YouTube Videos:")
        if videos:
            for video_url, thumbnail_url in videos:
                st.markdown(f'<a href="{video_url}" target="_blank"><img src="{thumbnail_url}" width="320"></a>', unsafe_allow_html=True)
        else:
            st.warning("No YouTube videos found.")
        
        # Fetch Free Courses
        courses = search_free_courses(topic)
        st.subheader("🎓 Free Courses:")
        if courses:
            for course_name, link in courses:
                st.markdown(f'<a href="{link}" target="_blank">{course_name}</a>', unsafe_allow_html=True)
        else:
            st.warning("No free courses found.")
