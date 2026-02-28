import re 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# -------PArt -1 -- load-->clean-->chunk
# loading the documents
file_path = 'iteration1/Constitution-of-Nepal_2072.pdf'
loader = PyPDFLoader(file_path)
pages = loader.load()
# print(pages)

# merge all the pages into one
constitution = "\n".join([page.page_content for page in pages])

# split by article so we can store by article 
# NFA(NOte for Abhyudaya): don't change this regex expr this is the structure of article 
article_pattern = r'(\n\d+\.\s+[A-Z][^\n:]+:.*?)(?=\n\d+\.\s+[A-Z][^\n:]+:|\Z)'  #this was the hardest part
articles = re.findall(article_pattern,constitution,re.DOTALL)


documents = []

# looks for the article numner in text 
# re.search find the first match of pattern Article\s+\d
# stores article num if found else unkwon 
for article in articles:
    match = re.search(r"Article\s+\d+", article)
    article_number = match.group() if match else "Unknown"


# wraps each article chunk into a document object
# page_content = the actual text of the article
# metadata={"article": article_number} = stores the article number in a separate field
    documents.append(
        Document(
            page_content=article.strip(),
            metadata={"article": article_number}
        )
    )

print(f'total article found {len(documents)}')
# splittter 
splitters = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)
docs = splitters.split_documents(documents)
print(f'total chunks after splitting: {len(docs)}')

# print(constitution[:1000])



# ---part 2 embedddings
# converted into 768 dim vector
# store alongside its text 
# article num is preserved
# indexed for similarity serach


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

# chroma vector to store constitution
vector = Chroma.from_documents(
    documents=docs,
    embedding= embeddings,
    persist_directory= 'chromaVectordb' # saves locally 
)

# save to disk 
vector.persist()
print("vector database saved sucessfully")
print(f'saved :{len(docs)}')



