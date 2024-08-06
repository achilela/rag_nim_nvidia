import streamlit as st
from io import StringIO
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from openai import OpenAI

# Function to get embeddings from NVIDIA NIM using OpenAI client
def get_embeddings(client, input_text, model):
    response = client.embeddings.create(
        input=[input_text],
        model=model,
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "NONE"}
    )
    return response.data[0].embedding

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_pdf):
    pdf_reader = PdfReader(uploaded_pdf)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    return pdf_text

# Function to retrieve content from a website
def retrieve_content_from_link(site_link):
    response = requests.get(site_link)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

# Sidebar for inputs
st.sidebar.title("RAG Streamlit WebApp")
api_key = st.sidebar.text_input("Enter NVIDIA Bearer Auth Token", type="password")
selected_model = st.sidebar.selectbox(
    "Select Embedding Model",
    ["nvidia/nv-embedqa-e5-v5", "other-model-1", "other-model-2"]  # Add more models as needed
)

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
site_link = st.sidebar.text_input("Enter Site Link for Retrieval")

# Main input area
st.title("RAG Streamlit WebApp using NVIDIA NIM")
input_text = st.text_area("Enter input text")

if st.button("Get Embeddings"):
    if api_key and input_text:
        client = OpenAI(api_key=api_key, base_url="https://integrate.api.nvidia.com/v1")
        embeddings = get_embeddings(client, input_text, selected_model)
        st.write("Embeddings:", embeddings)
    else:
        st.error("Please enter both the NVIDIA Bearer Auth Token and input text.")

# Process uploaded PDF
if uploaded_pdf is not None:
    pdf_text = extract_text_from_pdf(uploaded_pdf)
    st.write("PDF Content:")
    st.write(pdf_text)

# Handle site link retrieval
if site_link:
    site_content = retrieve_content_from_link(site_link)
    st.write("Site Content:")
    st.write(site_content)
