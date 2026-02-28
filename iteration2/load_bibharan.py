import os
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 999,
        chunk_overlap = 200,
        length_function = len,
    )

    split_docs = text_splitter.split_documents(documents)
    
    
    # Initializing local llama model through ollama
    llm = OllamaLLM(model="llama3.1")

    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose = True)

    #final summary
    summary = chain.invoke(split_docs)
    return summary['output_text']
    




if __name__ == "__main__":
    document_path = r"iteration2\smaple.pdf"

    if os.path.exists(document_path):
        summary = summarize_document(document_path)
        print("\n \n Aaja ko saramsha:")
        print(summary)
    else:
        print(f"Error: ${document_path} not found")
