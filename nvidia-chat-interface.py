import streamlit as st
import time
import json
from datetime import datetime

class ChatInterface:
    def __init__(self, embeddings_client):
        self.embeddings_client = embeddings_client

    def add_user_message(self, message):
        st.session_state.messages.append({"role": "user", "content": message})

    def add_ai_message(self, message):
        st.session_state.messages.append({"role": "assistant", "content": message})

    def get_ai_response(self, placeholder):
        # Simulate AI processing and streaming response
        full_response = ""
        for word in "I'm processing your request. Here's a simulated response from Methods Engineer B17...".split():
            full_response += word + " "
            placeholder.markdown(f"🚀 {full_response}▌")
            time.sleep(0.05)

        # Get embedding for the last user message
        last_user_message = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
        if last_user_message:
            embedding = self.embeddings_client.embed_query(last_user_message)
            # Here you would typically use this embedding to retrieve relevant information or generate a response
            # For this example, we'll just print the embedding
            print(f"🚀 Embedding for '{last_user_message}': {embedding[:5]}...")  # Print first 5 elements

        self.add_ai_message(full_response)
        return full_response

    def get_file_embedding(self, file_content):
        # For simplicity, we're just getting an embedding of the first 1000 characters
        # In a real application, you'd process the entire file content appropriately
        return self.embeddings_client.embed_query(file_content[:1000])

    def save_chat_history(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"methods_engineer_b17_chat_history_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(st.session_state.messages, f, indent=2)
        return filename
