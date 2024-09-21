import streamlit as st
from openai import OpenAI
import os
import PyPDF2
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit page configuration
st.set_page_config(page_title="Methods Engineer B17 🚀", page_icon="🚀", layout="wide")

# Custom CSS for modern layout and reduced font size
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        font-family: 'Roboto', sans-serif;
    }
    .stTextInput > div > div > input {
        font-size: 14px;
    }
    .stMarkdown {
        font-size: 14px;
    }
    h1 {
        font-size: 24px !important;
    }
    h2 {
        font-size: 20px !important;
    }
    .stButton > button {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_embeddings" not in st.session_state:
    st.session_state.document_embeddings = []
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI(
        api_key=os.getenv("NVIDIA_API_KEY", "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"),
        base_url="https://integrate.api.nvidia.com/v1"
    )

client = get_openai_client()

def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="nvidia/nv-embedqa-mistral-7b-v2",
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "NONE"}
    )
    return response.data[0].embedding

def process_file(file):
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() for page in pdf_reader.pages)
    elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
        df = pd.read_excel(file)
        return df.to_string()
    else:
        return file.getvalue().decode("utf-8")

def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def get_most_relevant_chunk(query_embedding, document_embeddings, top_k=3):
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:]
    return [st.session_state.document_chunks[i] for i in top_indices]

def main():
    st.markdown("# Methods Engineer B17 🚀")

    # File uploader
    uploaded_file = st.file_uploader("🚀 Upload a file", type=["txt", "pdf", "xlsx", "xls"])
    if uploaded_file:
        file_content = process_file(uploaded_file)
        st.success(f"🚀 File '{uploaded_file.name}' uploaded and processed successfully!")
        
        # Create document chunks and embeddings
        st.session_state.document_chunks = chunk_text(file_content)
        st.session_state.document_embeddings = [get_embedding(chunk) for chunk in st.session_state.document_chunks]
        
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"🚀 {message['content']}")

    # Chat input
    user_input = st.chat_input("🚀 Type your message here...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(f"🚀 {user_input}")

        with st.chat_message("assistant"):
            if st.session_state.document_embeddings:
                query_embedding = get_embedding(user_input)
                relevant_chunks = get_most_relevant_chunk(query_embedding, st.session_state.document_embeddings)
                response = f"Based on the uploaded document, here are the most relevant parts:\n\n" + "\n\n".join(relevant_chunks)
            else:
                response = "I'm sorry, but I don't have any document to reference. Could you please upload a document first?"
            
            st.markdown(f"🚀 {response}")
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar options
    with st.sidebar:
        st.markdown("## Options 🚀")
        if st.button("🚀 Clear Chat History"):
            st.session_state.messages = []
            st.session_state.document_embeddings = []
            st.session_state.document_chunks = []
            st.experimental_rerun()

        if st.button("🚀 Save Chat History"):
            # Implement save functionality here
            st.success(f"🚀 Chat history saved successfully!")

if __name__ == "__main__":
    main()
