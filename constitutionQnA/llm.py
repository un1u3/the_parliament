import os
import re 
os.environ["GPTJ4ALL_VERBOSE"] = "0"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from gpt4all import GPT4All

# load vector db 
# same same but diffrent 
# in load_constiution.py we made vector db here we use it 
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

vector = Chroma(
    persist_directory='chromaVectordb',
    embedding_function=embeddings
)

# load llm
# it will autodownload model if not exists
llm = GPT4All(
    model_name='Phi-3-mini-4k-instruct.Q4_0.gguf',
)
print("loaded")

# RAG 
# k = how many chunks to retrieve from ChromaDB.
def ask_constitution(question,k=3):
    results = vector.similarity_search(question,k=k)
    
    if not results:
        return "The Constituion of Nepal 2072 dosent adddress it directly"
    
    context = "\n\n---\n\n".join([
        f"[{r.metadata.get('article')} ----- {r.metadata.get('title', '')}]\n{r.page_content}"
        for r in results
    ])

    # promt enginner hawa ho, gardina vanda vanda garnu paryo fk
    prompt = f"""You are a constitutional lawyer in Nepal, speaking directly to a client.
                Speak like a real lawyer — plain english, confident, conversational, 1 paragraphs max.
                Only use the constitutional articles provided below as your legal basis.
                Never cite an article that is not in the context. Never fabricate law.
                If the constitution doesn't cover it, say so honestly.

                CONSTITUTIONAL CONTEXT:
                {context}

                CLIENT'S QUESTION: {question}

                YOUR RESPONSE AS THEIR LAWYER:"""

    with llm.chat_session():
        return llm.generate(
            prompt,
            max_tokens=500)

# more work is needed 
# you can call the ask_constitution() fn 
# parm can be string or list, list is preferable, 
# an array of question can also be passed
    