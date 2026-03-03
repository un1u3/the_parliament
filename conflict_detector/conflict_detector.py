from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def load_document():
    # using all same code from constitutionQnA 
    # might need a utils.py containing redundant fucntins 
    file_path = 'conflict_detector/Nepal.Social-Media-Bill_2025_Eng.pdf'
    loader = PyPDFLoader(file_path)
    doc = loader.load()
    full_text = "\n\n".join([doc.page_content for doc in docs])
    return full_text


def search_constitution():

    # load embeddings
    embeddings =  HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-mpnet-base-v2"
    )

    # load constitution from already existing chromaDb
    constitution_db = Chroma(
        persist_directory= '../chromaVectordb'
        embedding_function = embeddings

    )
