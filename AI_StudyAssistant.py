# PDF reader

import fitz
import io
import os
import easyocr
import numpy as np
from PIL import Image

pdf_input = input("Enter the location of the pdf : ").strip().strip('"')
pdf_path = pdf_input.replace("\\", "/")

# Create EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Open the PDF
doc = fitz.open(pdf_path)
all_text = ""

# Loop through each page
for page_num in range(len(doc)):
    #if 0 <= page_num <= 2:
        page = doc[page_num]
        page_text = f"[Page {page_num + 1}]\n"

        # ✅ Extract regular PDF text
        print(f"Extracting text from page no : {page_num + 1}")
        extracted_text = page.get_text().strip()
        if extracted_text:
            page_text += f"\n[Text Content]\n{extracted_text}\n"

        # ✅ Extract images + OCR
        image_list = page.get_images(full=True)
        if image_list:
            print(f"Extracting image text from page no : {page_num + 1}")
            for img_index, img in enumerate(image_list):
                img_id = img[0]
                base_image = doc.extract_image(img_id)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                np_image = np.array(image)
                image_text_list = reader.readtext(np_image, detail=0)
                image_text = "\n".join(image_text_list)

                page_text += f"\n[Image Content {img_index + 1}]\n{image_text.strip()}\n"

        all_text += page_text + "\n\n"


print("Text extract from pdf successfully ✅")

# Tokenizer

from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(chunk_size= 500 , chunk_overlap= 50)
chunks = splitter.split_text(all_text)
print(f"Total chunks created : {len(chunks)}")

# Document Converter

from langchain.schema import Document

documents = [Document(page_content= chunk)for chunk in chunks]
print(f"Total documents created : {len(documents)}")

# Embedding Model

from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2" )
print("Embedding model created successfully ✅")
# Vectorstorage

from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_documents(documents,embedding_model)
print("Embeddings stored in vector DB successfully ✅")
# print(vector_store)

# Retrievers

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

vector_retriever = vector_store.as_retriever(search_kwargs = {"k": 5})

bm25_retriever = BM25Retriever.from_documents(documents)


hybrid_retriever = EnsembleRetriever(
    retrievers = [vector_retriever, bm25_retriever],
    weights = [0.7 , 0.3]
)
print("Retrievers ready ✅")

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
print("LLM initialized successfully ✅")

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

First provide a detailed notes of 500 words on the topic, like this 
Notes:

Paragraph 1
Paragraph 2
Paragraph 3

Then create 5 MCQ questions, like this:
1) Question?
A. Option 1
B. Option 2
C. Option 3
D. Option 4

Then show the correct answers to the 5 MCQ questions like this :
Answers : 
1) A
2) B
3) C
4) D
5) A
"""
)
print("Custom prompt set successfully ✅")

# Retrival QA

from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type (
    llm = llm,
    retriever = hybrid_retriever,
    chain_type= "stuff",
    chain_type_kwargs = {"prompt":custom_prompt}
)
print("RetrievalQA set successfully ✅")

# User query section

def quer():
    query = input("Enter your query : ")

    # 🔍 Step 1: See what documents are retrieved
    #retrieved_docs = hybrid_retriever.get_relevant_documents(query)
    #print(f"\n📄 Retrieved {len(retrieved_docs)} documents:\n")

    #for i, doc in enumerate(retrieved_docs, 1):
    #    print(f"\n--- Document {i} ---\n{doc.page_content}\n")

    # ✅ Step 2: Call the QA chain
    raw_result = qa.invoke({"query": query})
    llm_output = raw_result["result"].strip()

    print("\n🤖 LLM Output:\n", llm_output)


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
