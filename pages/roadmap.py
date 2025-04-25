import streamlit as st
import os
import base64
from datetime import datetime
from fpdf import FPDF
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate   
from auth import save_user_roadmap, get_user_roadmaps,load_roadmap,save_roadmap# add this import
from markdown2 import markdown

def login_required():
    if not st.session_state.get("logged_in", False):
        st.warning("You must log in to access this page.")
        st.stop()


# Initialize model
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "API-KEY"


llm = init_chat_model("llama3-8b-8192", model_provider="groq")

# Prompt template
prompt = PromptTemplate.from_template("""You are an expert curriculum designer and technical educator. Your task is to create a detailed learning roadmap for the domain: {domain}.

Generate a comprehensive, topic-based (not time-based) roadmap that progresses from absolute basics to advanced expert level.

Format the roadmap with:

# 🚀 [Domain] Learning Roadmap

## 📚 Foundation Level
🔹 **Topic 1**: 
   - Key concepts
   - Practical applications
   - Resources (books/courses)
   
🔹 **Topic 2**: 
   - Key concepts
   - Practical applications
   - Resources

## 🏗️ Intermediate Level
🔸 **Topic 1**: 
   - Key concepts
   - Practical applications
   - Resources

## 🎯 Advanced Level
🔺 **Topic 1**: 
   - Key concepts
   - Practical applications
   - Resources

## 🏫 Expert Level
🌟 **Topic 1**: 
   - Key concepts
   - Practical applications
   - Resources

Include emojis to make it visually appealing and use clear section headers.""")

def generate_roadmap(domain):
    chain = prompt | llm
    response = chain.invoke({"domain": domain})
    return response.content

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download PDF</a>'


# UI
def main():
    st.set_page_config(page_title="Roadmap Generator", page_icon="🧭", layout="wide")

    st.title("🧭 Roadmap Generator")
    st.write("Enter a domain to generate a custom learning roadmap.")

    col1, col2 = st.columns([3, 1])
    with col1:
        domain = st.text_input("Enter your domain of interest:")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("✨ Generate Roadmap", use_container_width=True)

    if generate_btn:
        if not domain:
            st.warning("Please enter a domain.")
            return
        with st.spinner(f"Generating roadmap for {domain}..."):
            try:
                roadmap = generate_roadmap(domain)
                st.success("✅ Roadmap generated!")

                # Show roadmap
                with st.expander("📝 View Roadmap", expanded=True):
                    st.markdown(roadmap)

                # Save to DB
                save_user_roadmap(st.session_state.username, domain, roadmap)

                # Download option
                st.download_button(
                    label="📥 Download Markdown",
                    data=roadmap,
                    file_name=f"{domain.lower().replace(' ', '_')}_roadmap.md",
                    mime="text/markdown"
                )

                st.markdown("---")
                st.markdown(f"🗓️ Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.markdown("## 🗃️ Your Previous Roadmaps")
    previous = get_user_roadmaps(st.session_state.username)
    if previous:
        for dom, content, time in previous:
            with st.expander(f"📚 {dom} — {time}"):

                # Try loading progress JSON
                roadmap_data = load_roadmap(st.session_state.username, dom)

                if not roadmap_data:
                    # If no saved progress, parse from markdown into structure
                    roadmap_data = {}  # example fallback

                    current_level = None
                    for line in content.splitlines():
                        line = line.strip()
                        if line.startswith("##"):
                            current_level = line.replace("##", "").strip()
                            roadmap_data[current_level] = {}
                        elif line.startswith("🔹") or line.startswith("🔸") or line.startswith("🔺") or line.startswith("🌟"):
                            topic_title = line.split("**")[1] if "**" in line else line
                            topic_title = topic_title.replace("**", "").strip()
                            if current_level:
                                roadmap_data[current_level][topic_title] = {
                                    "description": "",
                                    "completed": False
                                }
                        elif "-" in line and current_level:
                            last_topic = list(roadmap_data[current_level].keys())[-1]
                            roadmap_data[current_level][last_topic]["description"] += line + "\n"

                st.subheader(f"📝 Progress Tracker for {dom}")
                for level, topics in roadmap_data.items():
                    st.markdown(f"### {level}")
                    for topic, data in topics.items():
                        key = f"{dom}_{level}_{topic}"
                        is_done = st.checkbox(f"{topic}: {data['description'].strip()}", value=data["completed"], key=key)
                        roadmap_data[level][topic]["completed"] = is_done

                if st.button(f"💾 Save Progress for {dom}"):
                    save_roadmap(st.session_state.username, dom, roadmap_data)
                    st.success(f"Progress for {dom} saved successfully!")
    else:
        st.info("No previous roadmaps found.")


if __name__ == "__main__":
    login_required()
    main()
