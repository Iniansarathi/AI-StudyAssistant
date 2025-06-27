# PDF reader

import fitz

pdf_path = r"C:\Users\MSI\Downloads\Iniansarathi Resume.pdf"
doc = fitz.open(pdf_path)
all_text = ""
for page in doc:
    page_text = page.get_text()
    all_text += page_text
print(all_text)

# Tokenizer

from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(chunk_size= 500 , chunk_overlap= 50)
chunks = splitter.split_text(all_text)
for chunk in chunks :
    print("************")
    print(chunk)
    print("************")
print(f"Total chunks created : {len(chunks)}")

# Document Converter

from langchain.schema import Document

documents = [Document(chunk)for chunk in chunks]
for docu in documents :
    print("#############")
    print(docu)
    print("#############")
print(f"Total documents created : {len(documents)}")

# Embedding Model

from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2" )
print("Embedding model created successfully")
# Vectorstorage

from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_documents(documents,embedding_model)
print("Embeddings stored in vector DB successfully")
print(vector_store)

# Retrievers

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

vector_retriever = vector_store.as_retriever(search_kwargs = {"k": 5})

bm25_retriever = BM25Retriever.from_documents(documents)


hybrid_retriever = EnsembleRetriever(
    retrievers = [vector_retriever, bm25_retriever],
    weights = [0.7 , 0.3]
)
print("✅ Retrievers ready.")

#Initialize LLM

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=api_key,
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.5
)
print("✅ LLM initialized successfully")

# Prompt Template

from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template= """You are a helpful assistant and expert in every subject.
If the context is empty or irrelevant, say "I couldn’t find relevant information."

Context:
{context}

Question:
{question}

First provide a detailed notes of 500 words on the topic, like this:
Notes:
...

Then create 5 MCQ questions, like this:
1) Question?
A. Option 1
B. Option 2
C. Option 3
D. Option 4
"""
)
print("Custom prompt set successfully")

# Retrival QA

from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type (
    llm = llm,
    retriever = hybrid_retriever,
    chain_type= "stuff",
    chain_type_kwargs = {"prompt":custom_prompt}
)
print("RetrievalQA set successfully")

# User query section

def quer ():
    query = input("Enter your query : ")
    raw_result = qa.invoke({"query" : query })
    llm_output = raw_result["result"].strip()
    print("LLM output : ",llm_output)

def yn():
    while True:
        a = input("Would you like to continue with a new query? [y/n]: ").strip().lower()
        if a == "y":
            return True
        elif a == "n":
            return False
        else:
            print("Please enter 'y' or 'n'.")


while True:
    quer()
    if not yn():
        break

print("Thank you 🙏")
