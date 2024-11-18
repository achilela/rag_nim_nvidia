import streamlit as st
import asyncio
import json
from datetime import datetime
from openai import AsyncOpenAI

class ChatInterface:
    def __init__(self, api_key, base_url):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def add_user_message(self, message):
        st.session_state.messages.append({"role": "user", "content": message})

    def add_ai_message(self, message):
        st.session_state.messages.append({"role": "assistant", "content": message})

    async def get_embedding(self, text):
        response = await self.client.embeddings.create(
            input=[text],
            model="nvidia/nv-embedqa-mistral-7b-v2",
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"}
        )
        return response.data[0].embedding

    async def stream_response(self, prompt):
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
            #await asyncio.sleep(0.01)

    async def get_ai_response(self, placeholder, prompt):
        embedding = await self.get_embedding(prompt)
        print(f"🚀 Embedding for '{prompt[:30]}...': {embedding[:5]}...")  # Print first 5 elements

        full_response = ""
        async for response in self.stream_response(prompt):
            placeholder.markdown(f"🚀 {response}▌")
            full_response = response  # Update full_response here
            
        placeholder.markdown(f"🚀 {full_response}")
        
        self.add_ai_message(full_response)  # Use final full_response
        return full_response

    async def get_file_embedding(self, file_content):
        # Only take the first 1000 characters to avoid excessive API calls or data processing
        return await self.get_embedding(file_content[:1000])

    def save_chat_history(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"methods_engineer_b17_chat_history_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(st.session_state.messages, f, indent=2)
        return filename
