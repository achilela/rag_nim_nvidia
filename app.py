import streamlit as st




from chat_interface import ChatInterface
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
import os
import PyPDF2
import pandas as pd
import io

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
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Initialize NVIDIA Embeddings client
@st.cache_resource
def get_embeddings_client():
    api_key = os.getenv("NVIDIA_API_KEY", "")
    return NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-mistral-7b-v2",
        api_key=api_key,
        truncate="NONE",
    )

client = get_embeddings_client()

# Initialize ChatInterface
chat_interface = ChatInterface(client)

def process_file(file):
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() for page in pdf_reader.pages)
    elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
        df = pd.read_excel(file)
        return df.to_string()
    else:
        return file.getvalue().decode("utf-8")

def main():
    st.markdown("# Methods Engineer B17 🚀")

    # File uploader
    uploaded_file = st.file_uploader("🚀 Upload a file", type=["txt", "pdf", "xlsx", "xls"])
    if uploaded_file:
        file_content = process_file(uploaded_file)
        st.success(f"🚀 File '{uploaded_file.name}' uploaded and processed successfully!")
        
        # Create vector store
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(file_content)
        st.session_state.vector_store = FAISS.from_texts(texts, client)
        
        # Create retrieval chain
        llm = OpenAI(temperature=0)
        st.session_state.retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm, st.session_state.vector_store.as_retriever(), return_source_documents=True
        )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"🚀 {message['content']}")

    # Chat input
    user_input = st.chat_input("🚀 Type your message here...")
    if user_input:
        chat_interface.add_user_message(user_input)
        with st.chat_message("user"):
            st.markdown(f"🚀 {user_input}")

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            if st.session_state.vector_store:
                chat_history = [(msg["content"], "") for msg in st.session_state.messages if msg["role"] == "user"]
                response = st.session_state.retrieval_chain({"question": user_input, "chat_history": chat_history})
                full_response = response['answer']
            else:
                full_response = chat_interface.get_ai_response(response_placeholder)
            response_placeholder.markdown(f"🚀 {full_response}")
            chat_interface.add_ai_message(full_response)

    # Sidebar options
    with st.sidebar:
        st.markdown("## Options 🚀")
        if st.button("🚀 Clear Chat History"):
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.experimental_rerun()

        if st.button("🚀 Save Chat History"):
            filename = chat_interface.save_chat_history()
            st.success(f"🚀 Chat history saved to {filename}")

if __name__ == "__main__":
    main()
