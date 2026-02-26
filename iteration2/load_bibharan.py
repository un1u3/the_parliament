import os
from langchain_community.document_loaders import PyPDFLoader

from langchain_ollama import OllamaLLM
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma

def summarize_document(file_path):
    loader = PyPDFLoader("sample.pdf")

    # Load the documents
    documents = loader.load()

    #  Split the documents into chunks
    pass


# Initializing local llama model through ollama
llm = OllamaLLM(model="llama3.1")


if __name__ == "__main__":
    document_path = "iteration2\smaple.pdf"

    if os.path.exists(document_path):
        summary = summarize_document(document_path)
        print("\n \n Aaja ko saramsha:")
        print(summary)
    else:
        print(f"Error: ${document_path} not found")
