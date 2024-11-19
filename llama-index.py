# chat_interface.py
import streamlit as st
from datetime import datetime
import json
from openai import AsyncOpenAI
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine

class EnhancedChatInterface:
    def __init__(self, api_key, base_url):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        if "messages" not in st.session_state:
            st.session_state.messages = []
        self.chat_engine = None

    def initialize_rag(self, content):
        documents = [Document(text=content)]
        index = VectorStoreIndex.from_documents(documents)
        memory = ChatMemoryBuffer.from_defaults(token_limit=4500)
        self.chat_engine = CondensePlusContextChatEngine.from_defaults(
            index.as_retriever(),
            memory=memory
        )

    async def get_embedding(self, text):
        response = await self.client.embeddings.create(
            input=[text],
            model="nvidia/nv-embedqa-mistral-7b-v2",
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"}
        )
        return response.data[0].embedding

    async def stream_response(self, prompt):
        if self.chat_engine:
            response = self.chat_engine.chat(prompt)
            yield str(response)
        else:
            response = self.client.chat.completions.create(
                model="nvidia/text-generation-mistral-7b",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            full_response = ""
            for chunk in response:
                if "content" in chunk.choices[0].delta:
                    chunk_content = chunk.choices[0].delta.content
                    full_response += chunk_content
                    yield full_response

    async def get_ai_response(self, placeholder, prompt):
        embedding = await self.get_embedding(prompt)
        full_response = ""
        
        async for response in self.stream_response(prompt):
            placeholder.markdown(f"🤖 {response}▌")
            full_response = response
            
        placeholder.markdown(f"🤖 {full_response}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })
        return full_response

    def save_chat_history(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(st.session_state.messages, f, indent=2)
        return filename

# app.py
import os
import asyncio
from dotenv import load_dotenv
import streamlit as st

async def main():
    load_dotenv()
    st.title("🚀 Twin - Methods Engineer with RAG 🚀")

    chat_interface = EnhancedChatInterface(
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url="https://integrate.api.nvidia.com/v1"
    )

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])
        if uploaded_file:
            content = process_file(uploaded_file)
            chat_interface.initialize_rag(content)
            st.success("Document processed")

        if st.button("Save Chat"):
            filename = chat_interface.save_chat_history()
            st.success(f"Chat saved to {filename}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            await chat_interface.get_ai_response(st.empty(), prompt)

if __name__ == "__main__":
    asyncio.run(main())