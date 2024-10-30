import streamlit as st
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import PyPDF2
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

token=""
model_name = "Meta-Llama-3.1-8B-Instruct"

endpoint = "https://models.inference.ai.azure.com"

def extract_text_from_pdf(uploaded_file):
    text = ""
    # Read the PDF directly from the uploaded file
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text



def process_pdf(uploaded_file):
    # Save the current position of the file pointer
    uploaded_file.seek(0)
    
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    # Reset the file pointer for any future operations
    uploaded_file.seek(0)
    
    # Process the text
    documents = [Document(page_content=pdf_text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = FastEmbedEmbeddings(embed_dim=1536)
    vector_store = FAISS.from_documents(texts, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    return retriever

def def_ask_question(endpoint, token, model_name, retriever, question):

    try:
        # Initialize the Azure ChatCompletionsClient
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token)
        )

        # Get relevant documents from the retriever
        context_docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in context_docs])

        # Generate response using Azure OpenAI
        response = client.complete(
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant"
                },
                {
                    "role": "user", 
                    "content": f'Based on the following context:\n \n{context} \n\n Answer this question :{question}'
                }
            ],
            model=model_name,
            temperature=0.3,
            max_tokens=500,
            top_p=1.0
        )

        # Extract and return the response
        return response.choices[0].message.content if response.choices else "No response available"

    except Exception as e:
        return f"An error occurred: {str(e)}"




# Streamlit app
st.title("PDF Q&A")

# File upload
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    # Process the uploaded PDF
    retriever = process_pdf(uploaded_file)

    # User input for question
    user_question = st.text_input("Ask your question:")

    if user_question:
        # Generate and display the answer
        answer = def_ask_question(
            retriever=retriever,
            question=user_question
        )
        st.write("Answer:", answer)

