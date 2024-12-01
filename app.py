import os
import asyncio
import base64
import streamlit as st
import PyPDF2
import pandas as pd
import requests
from openai import OpenAI
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
import re

def clean_response_text(text):
    """Clean and format the response text for better readability"""
    # Remove escape sequences
    text = text.replace('\\n', '\n').replace('\\r', '').replace('\\t', ' ')
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Clean up newlines
    text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
    
    # Add proper spacing after periods if missing
    text = re.sub(r'\.(?=[A-Z])', '. ', text)
    
    return text

# Load environment variables
load_dotenv()

# Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "nvapi-")
VISION_API_URL = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions"

class NVIDIAInterface:
    def __init__(self):
        self.text_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )
        self.vision_headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "text/event-stream"
        }

    def process_text(self, prompt, stream=True):
        return self.text_client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=stream
        )

    def process_image(self, image_data, prompt):
        image_b64 = base64.b64encode(image_data).decode()
        
        payload = {
            "model": 'meta/llama-3.2-90b-vision-instruct',
            "messages": [
                {
                    "role": "user",
                    "content": f'{prompt} <img src="data:image/png;base64,{image_b64}" />'
                }
            ],
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True
        }
        
        return requests.post(
            VISION_API_URL, 
            headers=self.vision_headers, 
            json=payload, 
            stream=True
        )

def process_file(file):
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() for page in pdf_reader.pages)
    elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
        df = pd.read_excel(file)
        return df.to_string()
    else:
        return file.getvalue().decode("utf-8")

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mode" not in st.session_state:
        st.session_state.mode = "text"

def create_sidebar():
    with st.sidebar:
        st.header("Settings")
        
        # Mode selection
        st.session_state.mode = st.radio(
            "Select Mode",
            ["text", "vision"],
            captions=["Process documents and text", "Analyze images"]
        )
        
        if st.session_state.mode == "text":
            st.header("Document Upload")
            uploaded_file = st.file_uploader(
                "Upload a file",
                type=["txt", "pdf", "xlsx", "xls"]
            )
            
            if uploaded_file:
                file_content = process_file(uploaded_file)
                st.success(f"File '{uploaded_file.name}' processed successfully!")
                st.session_state.file_content = file_content
                
        else:  # vision mode
            st.header("Image Upload")
            uploaded_image = st.file_uploader(
                "Upload an image",
                type=["png", "jpg", "jpeg"]
            )
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                st.session_state.image_data = uploaded_image.getvalue()
        
        # Common controls
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.pop('file_content', None)
            st.session_state.pop('image_data', None)
        
        if st.button("Save Chat History"):
            st.download_button(
                "Download Chat",
                "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages]),
                "chat_history.txt"
            )

def process_text_chat(nvidia_interface, prompt):
    full_prompt = prompt
    if 'file_content' in st.session_state:
        full_prompt = f"Based on the following document: {st.session_state.file_content[:1000]}... Please answer: {prompt}"
    
    completion = nvidia_interface.process_text(full_prompt)
    
    full_response = ""
    message_placeholder = st.empty()
    
    for chunk in completion:
        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
            message_placeholder.markdown(full_response + "▌")
    
    message_placeholder.markdown(full_response)
    return full_response

def process_vision_chat(nvidia_interface, prompt):
    if 'image_data' not in st.session_state:
        st.warning("Please upload an image first!")
        return None
    
    response = nvidia_interface.process_image(st.session_state.image_data, prompt)
    
    full_response = ""
    message_placeholder = st.empty()
    
    for line in response.iter_lines():
        if line:
            try:
                data = line.decode("utf-8")
                if "content" in data:
                    # Extract content and clean up the response
                    content = data.split('"content":"')[1].split('"')[0]
                    # Replace escaped newlines with actual newlines
                    content = content.replace('\\n', '\n')
                    # Remove other common escape sequences
                    content = content.replace('\\r', '')
                    content = content.replace('\\t', ' ')
                    # Replace multiple newlines with single newlines
                    content = '\n'.join(line.strip() for line in content.split('\n') if line.strip())
                    full_response = content
                    # Format the response for better readability
                    formatted_response = full_response.replace('*', '**')  # Make important points bold
                    message_placeholder.markdown(formatted_response)
            except Exception as e:
                continue
    
    return full_response

async def main():
    st.title("🚀 Twin - Methods Engineer with Vision 🚀")
    
    init_session_state()
    create_sidebar()
    nvidia_interface = NVIDIAInterface()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input and processing
    if prompt := st.chat_input("What's your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            if st.session_state.mode == "text":
                response = process_text_chat(nvidia_interface, prompt)
            else:
                response = process_vision_chat(nvidia_interface, prompt)
            
            if response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

if __name__ == "__main__":
    asyncio.run(main())
