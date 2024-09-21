import streamlit as st
from openai import AsyncOpenAI
import os
import PyPDF2
import pandas as pd
import asyncio
import time

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

# Initialize AsyncOpenAI client
@st.cache_resource
def get_openai_client():
    return AsyncOpenAI(
        api_key=os.getenv("NVIDIA_API_KEY", "nvapi-xHX1"),
        base_url="https://integrate.api.nvidia.com/v1"
    )

client = get_openai_client()

def process_file(file):
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() for page in pdf_reader.pages)
    elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
        df = pd.read_excel(file)
        return df.to_string()
    else:
        return file.getvalue().decode("utf-8")

async def stream_response(prompt):
    response = await client.chat.completions.create(
        model="nvidia/text-generation-mistral-7b",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    full_response = ""
    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            yield full_response
        await asyncio.sleep(0.01)

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

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.pop('file_content', None)

    # Main chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What's your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            if 'file_content' in st.session_state:
                full_prompt = f"Based on the following document: {st.session_state.file_content[:1000]}... Please answer: {prompt}"
            else:
                full_prompt = prompt
            
            async for response in stream_response(full_prompt):
                message_placeholder.markdown(response + "▌")
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    asyncio.run(main())
