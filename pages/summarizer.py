import streamlit as st
import os
import base64
import time
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from io import StringIO
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing_extensions import List, TypedDict, Annotated


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langgraph.graph import START, StateGraph
from langchain.chat_models import init_chat_model
# from langchain_mistralai import MistralAIEmbeddings
from langchain_core.prompts import PromptTemplate
import asyncio

def login_required():
    if not st.session_state.get("logged_in", False):
        st.warning("You must log in to access this page.")
        st.stop()

# from app import authenticated_page

if 'success' not in st.session_state:
    st.session_state.success = False
if 'retriever' not in st.session_state:
    st.session_state.retriever = None


# if not os.environ.get("MISTRAL_API_KEY"):
#     os.environ["MISTRAL_API_KEY"] = "API-KEY"
#     os.environ["HF_TOKEN"] = "API-KEY"

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "API-KEY"

# llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
llm = init_chat_model("llama3-8b-8192", model_provider="groq")
# prompt = hub.pull("rlm/rag-prompt")
template = """Use the following pieces of context to summarize answer in 1000 words for the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Helpful Answer:"""

prompt = PromptTemplate.from_template(template)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embeddings = MistralAIEmbeddings(model="mistral-embed")

async def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


async def get_text_chunks(text):
    # text_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="standard_deviation")
    # docs = text_splitter.create_documents(texts=[text])
    # return docs

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300,add_start_index=True)
    docs = text_splitter.split_text(text)
    return docs



async def get_retriever(text_chunks):
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
    )

    uuids = [str(uuid4()) for _ in range(len(text_chunks))]

    vector_store.add_texts(text_chunks, ids=uuids)
    return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})


class Search(TypedDict):
    """Search query."""
    query: Annotated[str, ..., "Search query to run."]
    

class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str


def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}


def retrieve(state: State):
    query = state["query"]
    retrieved_docs = st.session_state.retriever.invoke(query["query"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# @authenticated_page
async def summary():
    st.write("---")
    st.title("📁 PDF File's Section")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    if st.button("Submit & Process") and pdf_docs:
        with st.spinner("Processing..."):
            raw_text = await get_pdf_text(pdf_docs=pdf_docs)
            text_chunks = await get_text_chunks(raw_text)
            st.session_state.retriever = await get_retriever(text_chunks=text_chunks)
            st.session_state.success = True
            st.success(f"Processed {len(pdf_docs)} PDF(s) into {len(text_chunks)} chunks!")
            
    if st.session_state.success:
        user_text = st.text_area("Enter text to get your answer")
        if user_text and st.button("generate Response"):
            with st.spinner("Creating Summary..."):
                create_summary(user_text)

                
def create_summary(topic):
    # results = st.session_state.retriever.invoke(topic)
    # for res in results:
    #     st.write(f"* {res} [{res.metadata}]")

    graph_builder = StateGraph(State)
    graph_builder.add_node("analyze_query", analyze_query)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)    


    graph_builder.add_edge("analyze_query", "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.set_entry_point("analyze_query")
    graph = graph_builder.compile()
    result = graph.invoke({"question": topic})
    st.write("### Summarize:")
    st.write(result["answer"])


login_required()
asyncio.run(summary())
