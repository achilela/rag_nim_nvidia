import streamlit as st
import requests
import PyPDF2
from io import StringIO

# Function to get embeddings from NVIDIA NIM
def get_embeddings(model, input_text, token):
    url = "https://integrate.api.nvidia.com/v1/embeddings"
    payload = {
        "model": model,
        "input": input_text,
        "encoding_format": "float",
        "truncate": "NONE"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

# Sidebar for inputs
st.sidebar.title("RAG Streamlit WebApp")
auth_token = st.sidebar.text_input("Enter NVIDIA Bearer Auth Token", type="password")
selected_model = st.sidebar.selectbox(
    "Select Embedding Model",
    ["baai/bge-m3", "other-model-1", "other-model-2"]  # Add more models as needed
)

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
site_link = st.sidebar.text_input("Enter Site Link for Retrieval")

# Main input area
st.title("RAG Streamlit WebApp using NVIDIA NIM")
input_text = st.text_area("Enter input text")

if st.button("Get Embeddings"):
    if auth_token and input_text:
        embeddings = get_embeddings(selected_model, input_text, auth_token)
        st.write("Embeddings:", embeddings)
    else:
        st.error("Please enter both the NVIDIA Bearer Auth Token and input text.")

# Process uploaded PDF
if uploaded_pdf is not None:
    pdf_reader = PyPDF2.PdfFileReader(uploaded_pdf)
    pdf_text = ""
    for page in range(pdf_reader.numPages):
        pdf_text += pdf_reader.getPage(page).extract_text()
    st.write("PDF Content:")
    st.write(pdf_text)

# Handle site link (for future implementation)
if site_link:
    st.write(f"Retrieving content from: {site_link}")
    # Add your retrieval logic here
