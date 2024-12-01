import os
import asyncio
import streamlit as st
import PyPDF2
import pandas as pd
from openai import OpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()  # This will load the .env file where NVIDIA_API_KEY is expected to be defined

def get_chat_interface():
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY", "nvapi-")
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
    st.title("🚀 Twin - Methods Engineer 🚀")

    # Sidebar
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "xlsx", "xls"])
        if uploaded_file:
            file_content = process_file(uploaded_file)
            st.success(f"File '{uploaded_file.name}' processed successfully!")
            st.session_state.file_content = file_content
            with st.spinner('Creating file embedding...'):
                file_embedding = chat_interface.embeddings.create(input=file_content)
            st.session_state.file_embedding = file_embedding.data[0].embedding

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.pop('file_content', None)
            st.session_state.pop('file_embedding', None)

        if st.button("Save Chat History"):
            # Note: Saving chat history might require additional logic or external storage
            st.success("Chat history saved functionality not implemented yet.")

    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What's your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_prompt = prompt
            if 'file_content' in st.session_state:
                full_prompt = f"Based on the following document: {st.session_state.file_content[:1000]}... Please answer: {prompt}"
            
            completion = chat_interface.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
                stream=True
            )
            
            full_response = ""
            for chunk in completion:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    asyncio.run(main())
