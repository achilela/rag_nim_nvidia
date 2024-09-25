import streamlit as st
import os
import PyPDF2
import pandas as pd
import asyncio
from chat_interface import ChatInterface

# Streamlit page configuration
st.set_page_config(page_title="Methods Engineer B17 🚀", page_icon="🚀")

# Custom CSS for modern layout
st.markdown("""
<style>
    .stApp {
        max-width: 1000px;
        margin: 0 auto;
        font-family: 'Roboto', sans-serif;
    }
    .stSidebar {
        background-color: #f0f2f6;
        padding: 2rem 1rem;
    }
    .stSidebar .stButton > button {
        width: 100%;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e6f3ff;
        align-items: flex-end;
    }
    .assistant-message {
        background-color: #f0f0f0;
        align-items: flex-start;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize ChatInterface
@st.cache_resource
def get_chat_interface():
    return ChatInterface(
        api_key=os.getenv("NVIDIA_API_KEY", "nvapi-"),
        base_url="https://integrate.api.nvidia.com/v1"
    )

chat_interface = get_chat_interface()

def process_file(file):
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() for page in pdf_reader.pages)
    elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
        df = pd.read_excel(file)
        return df.to_string()
    else:
        return file.getvalue().decode("utf-8")

async def main():
    st.title("Methods Engineer B17 🚀")

    # Sidebar
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "xlsx", "xls"])
        if uploaded_file:
            file_content = process_file(uploaded_file)
            st.success(f"File '{uploaded_file.name}' processed successfully!")
            st.session_state.file_content = file_content
            file_embedding = await chat_interface.get_file_embedding(file_content)
            st.session_state.file_embedding = file_embedding

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.pop('file_content', None)
            st.session_state.pop('file_embedding', None)

        if st.button("Save Chat History"):
            filename = chat_interface.save_chat_history()
            st.success(f"Chat history saved to {filename}")

    # Main chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What's your question?"):
        chat_interface.add_user_message(prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            if 'file_content' in st.session_state:
                full_prompt = f"Based on the following document: {st.session_state.file_content[:1000]}... Please answer: {prompt}"
            else:
                full_prompt = prompt
            
            response = await chat_interface.get_ai_response(message_placeholder, full_prompt)

if __name__ == "__main__":
    asyncio.run(main())
