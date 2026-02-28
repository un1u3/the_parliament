from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

# for langchan version dep
try:
    from langchain_classic.chains.summarize import load_summarize_chain
except ModuleNotFoundError:
    from langchain.chains.summarize import load_summarize_chain

from langchain_ollama import OllamaLLM
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma

def summarize_document(file_path):
    loader = PyPDFLoader(file_path)

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
    base_dir = Path(__file__).resolve().parent
    document_path = base_dir / "sample.pdf"

    if document_path.exists():
        summary = summarize_document(str(document_path))
        print("\n \n Aaja ko saramsha:")
        print(summary)
    else:
        print(f"Error: {document_path} not found")
